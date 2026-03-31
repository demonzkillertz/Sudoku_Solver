import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';
import 'package:image/image.dart' as dart_img;
import 'package:image_picker/image_picker.dart';

import '../models/sudoku_board.dart';
import '../services/native_bridge.dart';
import '../services/ocr_engine.dart';
import '../services/sudoku_solver.dart';
import '../utils/overlay_data_bridge.dart';
import 'test_pipeline_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, this.initializeModel = true});

  final bool initializeModel;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final SudokuBoard _board = SudokuBoard.empty();
  final SudokuBoard _solvedBoard = SudokuBoard.empty();
  final NativeBridge _nativeBridge = NativeBridge();

  SudokuOcrEngine? _ocrEngine;
  bool _isModelLoaded = false;
  bool _isBusy = false;
  bool _isOverlayShowing = false;
  bool _captureReady = false;
  String _status = 'Loading model...';
  StreamSubscription<dynamic>? _overlayListenerSub;
  List<double>? _lastSolvedBoardRect;
  Uint8List? _previewImageBytes;
  double? _previewImageWidth;
  double? _previewImageHeight;
  double? _lastCaptureWidth;
  double? _lastCaptureHeight;
  int _lastCaptureModifiedMs = 0;
  int _lastCaptureSizeBytes = 0;

  // Grid calibration (matches main.py defaults)
  final TextEditingController _leftCtrl = TextEditingController(text: '70');
  final TextEditingController _topCtrl = TextEditingController(text: '420');
  final TextEditingController _cellSizeCtrl = TextEditingController(
    text: '120',
  );

  // Number selector coordinates (from main.py)
  final Map<int, List<int>> _numberCoords = {
    1: [110, 1900],
    2: [220, 1900],
    3: [330, 1900],
    4: [440, 1900],
    5: [550, 1900],
    6: [660, 1900],
    7: [770, 1900],
    8: [880, 1900],
    9: [990, 1900],
  };

  @override
  void initState() {
    super.initState();
    _listenToOverlayEvents();
    if (widget.initializeModel) {
      _initializeOcrEngine();
    } else {
      _isModelLoaded = true;
      _status = 'Ready (model init skipped).';
    }
  }

  Future<void> _initializeOcrEngine() async {
    try {
      _ocrEngine = SudokuOcrEngine();
      await _ocrEngine!.loadModel();
      setState(() {
        _isModelLoaded = true;
        _status = 'Model loaded! Ready to solve.';
      });
    } catch (e) {
      setState(() => _status = 'Model error: $e');
      print('Model loading error: $e');
    }
  }

  @override
  void dispose() {
    _overlayListenerSub?.cancel();
    _nativeBridge.stopOverlayService();
    _ocrEngine?.dispose();
    _leftCtrl.dispose();
    _topCtrl.dispose();
    _cellSizeCtrl.dispose();
    super.dispose();
  }

  // ==================== SCAN MODES ====================

  Future<void> _solveFromCamera() async {
    await _solveFromSource(ImageSource.camera, 'Camera');
  }

  Future<void> _solveFromGallery() async {
    await _solveFromSource(ImageSource.gallery, 'Gallery');
  }

  Future<void> _solveFromSource(ImageSource source, String modeName) async {
    if (!_isModelLoaded) {
      await _publishStatus('Model not loaded yet!');
      return;
    }

    setState(() {
      _isBusy = true;
      _status = 'Capturing from $modeName...';
    });

    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: source,
        imageQuality: 96,
        maxWidth: 2200,
        maxHeight: 2200,
      );

      if (pickedFile == null) {
        await _publishStatus('No image selected');
        return;
      }

      final bytes = await File(pickedFile.path).readAsBytes();
      final solved = await _processAndSolveImageBytes(
        bytes,
        sourceLabel: modeName,
      );
      if (!solved) {
        return;
      }

      await _publishStatus(
        'Solved from $modeName. Preview updated with solution overlay.',
      );
    } catch (e) {
      await _publishStatus('Error: $e');
    } finally {
      setState(() => _isBusy = false);
    }
  }

  Future<void> _publishStatus(String message) async {
    if (mounted) {
      setState(() => _status = message);
    }
    OverlayDataBridge.status = message;
    if (_isOverlayShowing) {
      await FlutterOverlayWindow.shareData(
        jsonEncode(OverlayDataBridge.toJson()),
      );
    }
  }

  Future<String?> _captureScreenWithRetry() async {
    const maxAttempts = 3;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      await _publishStatus(
        'Capturing screen (attempt $attempt/$maxAttempts)...',
      );

      final path = await _nativeBridge.captureScreen();
      if (path == null || path.isEmpty) {
        await Future.delayed(const Duration(milliseconds: 240));
        continue;
      }

      final captureFile = File(path);
      for (int waitStep = 0; waitStep < 10; waitStep++) {
        if (await captureFile.exists()) {
          final stat = await captureFile.stat();
          final modifiedMs = stat.modified.millisecondsSinceEpoch;
          final looksFresh =
              _lastCaptureModifiedMs == 0 ||
              modifiedMs > _lastCaptureModifiedMs ||
              stat.size != _lastCaptureSizeBytes;

          if (stat.size > 2048 && looksFresh) {
            _lastCaptureModifiedMs = modifiedMs;
            _lastCaptureSizeBytes = stat.size;
            return path;
          }
        }
        await Future.delayed(const Duration(milliseconds: 140));
      }

      await Future.delayed(const Duration(milliseconds: 280));
    }

    return null;
  }

  Future<void> _shareOverlayDataIfNeeded() async {
    if (!_isOverlayShowing) {
      return;
    }
    await FlutterOverlayWindow.shareData(
      jsonEncode(OverlayDataBridge.toJson()),
    );
  }

  Future<bool> _processAndSolveImageBytes(
    Uint8List rawBytes, {
    required String sourceLabel,
  }) async {
    if (_ocrEngine == null) {
      await _publishStatus('OCR engine is not ready yet.');
      return false;
    }

    await _publishStatus('Processing $sourceLabel image...');

    final decodedRaw = dart_img.decodeImage(rawBytes);
    if (decodedRaw == null) {
      await _publishStatus('Could not decode $sourceLabel image.');
      return false;
    }

    const maxOcrInputSide = 2200;
    const maxRawPathBytes = 8 * 1024 * 1024;

    var orientedImage = dart_img.bakeOrientation(decodedRaw);
    final orientedLongSideBeforeResize =
        orientedImage.width > orientedImage.height
        ? orientedImage.width
        : orientedImage.height;
    final resizedForMemory = orientedLongSideBeforeResize > maxOcrInputSide;

    if (resizedForMemory) {
      await _publishStatus('Optimizing large $sourceLabel image for OCR...');
      final scale = maxOcrInputSide / orientedLongSideBeforeResize;
      orientedImage = dart_img.copyResize(
        orientedImage,
        width: (orientedImage.width * scale).round(),
        height: (orientedImage.height * scale).round(),
      );
    }

    final orientedBytesLossless = Uint8List.fromList(
      dart_img.encodePng(orientedImage),
    );

    _previewImageBytes = orientedBytesLossless;
    _previewImageWidth = orientedImage.width.toDouble();
    _previewImageHeight = orientedImage.height.toDouble();
    _lastCaptureWidth = _previewImageWidth;
    _lastCaptureHeight = _previewImageHeight;

    final shouldUseRawPathFirst =
        !resizedForMemory && rawBytes.length <= maxRawPathBytes;

    Map<String, dynamic>? extractResult;
    if (shouldUseRawPathFirst) {
      // Keep runtime behavior aligned with the working pipeline test path.
      extractResult = await _ocrEngine!.processImage(rawBytes);
    }

    // Some camera/screen providers output edge-case encodings; retry with
    // normalized lossless bytes before failing detection.
    extractResult ??= await _ocrEngine!.processImage(
      orientedBytesLossless,
      alreadyNormalized: true,
    );

    // Final fallback for edge cases where normalized bytes fail but raw can pass.
    if (extractResult == null && !shouldUseRawPathFirst) {
      extractResult = await _ocrEngine!.processImage(rawBytes);
    }

    final detectedBoard = extractResult?['board'] as List<List<int>>?;
    final boardRect = extractResult?['rect'] as List<double>?;
    final lowConfidenceDrops =
        (extractResult?['lowConfidenceDrops'] as int?) ?? 0;

    if (lowConfidenceDrops > 0) {
      await _publishStatus(
        'Filtered $lowConfidenceDrops low-confidence OCR digits.',
      );
    }

    if (detectedBoard == null) {
      await _publishStatus(
        'Board not found. Keep the Sudoku board fully visible and retry.',
      );
      return false;
    }

    _board.cells = detectedBoard;
    _printBoard('Detected ($sourceLabel)', _board.cells);

    final boardForSolve = await _prepareBoardForSolve(_board.cells);
    if (boardForSolve == null) {
      return false;
    }
    _board.cells = boardForSolve;
    _printBoard('Prepared ($sourceLabel)', _board.cells);

    await _publishStatus('Solving puzzle...');

    final solvedCells = _cloneBoardCells(_board.cells);
    final solved = SudokuSolver.solveBoard(solvedCells);

    if (!solved) {
      await _publishStatus('No solution found!');
      return false;
    }

    _solvedBoard.cells = solvedCells;
    _lastSolvedBoardRect = boardRect;

    OverlayDataBridge.originalBoard = _cloneBoardCells(_board.cells);
    OverlayDataBridge.solvedBoard = _cloneBoardCells(_solvedBoard.cells);
    OverlayDataBridge.status = 'Solved from $sourceLabel';
    await _shareOverlayDataIfNeeded();

    if (mounted) {
      setState(() {});
    }

    return true;
  }

  Future<void> _solveFromScreen({bool autoFill = false}) async {
    if (_isBusy) return;
    if (!_isModelLoaded || _ocrEngine == null) {
      await _publishStatus('Model not loaded yet!');
      return;
    }

    setState(() {
      _isBusy = true;
    });
    await _publishStatus('Preparing screen capture...');

    final isCaptureReady = await _prepareCaptureService();
    if (!isCaptureReady) {
      if (mounted) {
        setState(() => _isBusy = false);
      }
      return;
    }

    await _setOverlayClickThrough(true);
    try {
      final String? path = await _captureScreenWithRetry();
      if (path == null) {
        await _publishStatus(
          'Capture failed. Keep the Sudoku app visible and try again.',
        );
        return;
      }

      await _publishStatus('Processing image...');

      final bytes = await File(path).readAsBytes();
      if (bytes.isEmpty) {
        await _publishStatus('Captured image is empty.');
        return;
      }

      final solved = await _processAndSolveImageBytes(
        bytes,
        sourceLabel: 'Screen',
      );
      if (!solved) {
        return;
      }

      if (!autoFill) {
        await _publishStatus(
          'Board scanned and solved. Tap Fill Board to apply.',
        );
        return;
      }

      await _publishStatus('Applying solved board...');
      final didFill = await _autoTapSolution(
        explicitRect: _lastSolvedBoardRect,
        imageWidth: _lastCaptureWidth,
        imageHeight: _lastCaptureHeight,
      );

      if (didFill) {
        await _publishStatus('Solved and filled successfully!');
      } else {
        await _publishStatus('Solved, but fill did not complete.');
      }
    } catch (e) {
      await _publishStatus('Error: $e');
    } finally {
      await _setOverlayClickThrough(false);
      if (mounted) {
        setState(() => _isBusy = false);
      }
    }
  }

  List<List<int>> _cloneBoardCells(List<List<int>> source) {
    return List.generate(9, (row) => List<int>.from(source[row]));
  }

  int _countFilledCells(List<List<int>> board) {
    int count = 0;
    for (final row in board) {
      for (final value in row) {
        if (value != 0) {
          count++;
        }
      }
    }
    return count;
  }

  bool _isPlacementValid(List<List<int>> board, int row, int col, int value) {
    for (int c = 0; c < 9; c++) {
      if (board[row][c] == value) {
        return false;
      }
    }
    for (int r = 0; r < 9; r++) {
      if (board[r][col] == value) {
        return false;
      }
    }

    final boxRow = (row ~/ 3) * 3;
    final boxCol = (col ~/ 3) * 3;
    for (int r = boxRow; r < boxRow + 3; r++) {
      for (int c = boxCol; c < boxCol + 3; c++) {
        if (board[r][c] == value) {
          return false;
        }
      }
    }
    return true;
  }

  List<List<int>> _clearConflictingDigits(List<List<int>> source) {
    final repaired = _cloneBoardCells(source);

    for (int row = 0; row < 9; row++) {
      for (int col = 0; col < 9; col++) {
        final value = repaired[row][col];
        if (value == 0) {
          continue;
        }

        repaired[row][col] = 0;
        final canKeep = _isPlacementValid(repaired, row, col, value);
        if (canKeep) {
          repaired[row][col] = value;
        }
      }
    }

    return repaired;
  }

  Future<List<List<int>>?> _prepareBoardForSolve(
    List<List<int>> detected,
  ) async {
    final initialValidation = SudokuSolver.validateBoard(detected);
    if (initialValidation['valid'] == true) {
      return _cloneBoardCells(detected);
    }

    final initialError = (initialValidation['error'] ?? '').toString();
    final isDuplicateIssue = initialError.toLowerCase().contains('duplicate');
    if (!isDuplicateIssue) {
      await _publishStatus('Invalid board: $initialError');
      return null;
    }

    await _publishStatus('OCR conflict detected. Repairing board...');
    final repaired = _clearConflictingDigits(detected);
    final removedCount =
        _countFilledCells(detected) - _countFilledCells(repaired);

    final repairedValidation = SudokuSolver.validateBoard(repaired);
    if (repairedValidation['valid'] != true) {
      await _publishStatus(
        'Invalid board: ${repairedValidation['error']}. Try a cleaner capture.',
      );
      return null;
    }

    await _publishStatus(
      'Repaired OCR conflicts (cleared $removedCount cells).',
    );
    return repaired;
  }

  // ==================== OVERLAY & AUTO-TAP ====================

  String _extractKnownOverlayAction(String raw) {
    final message = raw.toUpperCase();
    if (message.contains('SCAN_BOARD_REQUEST')) {
      return 'SCAN_BOARD_REQUEST';
    }
    if (message.contains('FILL_BOARD_REQUEST')) {
      return 'FILL_BOARD_REQUEST';
    }
    if (message.contains('SCAN_REQUEST')) {
      return 'SCAN_REQUEST';
    }
    return '';
  }

  String _extractOverlayAction(dynamic data) {
    if (data == null) {
      return '';
    }

    if (data is Map) {
      const keys = ['action', 'type', 'event', 'message', 'command', 'cmd'];
      for (final key in keys) {
        final value = data[key];
        if (value is String && value.trim().isNotEmpty) {
          final action = _extractKnownOverlayAction(value.trim());
          if (action.isNotEmpty) {
            return action;
          }
        }
      }
    }

    final raw = data.toString().trim();
    if (raw.isEmpty) {
      return '';
    }

    try {
      final decoded = jsonDecode(raw);
      if (decoded is String && decoded.trim().isNotEmpty) {
        final action = _extractKnownOverlayAction(decoded.trim());
        if (action.isNotEmpty) {
          return action;
        }
      }
      if (decoded is Map) {
        const keys = ['action', 'type', 'event', 'message', 'command', 'cmd'];
        for (final key in keys) {
          final value = decoded[key];
          if (value is String && value.trim().isNotEmpty) {
            final action = _extractKnownOverlayAction(value.trim());
            if (action.isNotEmpty) {
              return action;
            }
          }
        }
      }
    } catch (_) {
      // Not JSON payload, continue with plain text handling.
    }

    if ((raw.startsWith('"') && raw.endsWith('"')) ||
        (raw.startsWith("'") && raw.endsWith("'"))) {
      return _extractKnownOverlayAction(
        raw.substring(1, raw.length - 1).trim(),
      );
    }

    return _extractKnownOverlayAction(raw);
  }

  void _listenToOverlayEvents() {
    if (_overlayListenerSub != null) {
      return;
    }

    _overlayListenerSub = FlutterOverlayWindow.overlayListener.listen((
      data,
    ) async {
      if (data == null) return;

      final message = _extractOverlayAction(data);
      if (message.isEmpty) {
        return;
      }

      debugPrint('Overlay action: $message');

      if (message.contains('SCAN_BOARD_REQUEST') ||
          message.contains('SCAN_REQUEST')) {
        if (_isBusy) {
          await _publishStatus('Already processing. Please wait...');
          return;
        }
        await _publishStatus('Capture request received...');
        await _solveFromScreen(autoFill: false);
        return;
      }

      if (message.contains('FILL_BOARD_REQUEST')) {
        if (_isBusy) {
          await _publishStatus('Busy. Retry Fill Board after current task.');
          return;
        }
        await _fillSolvedBoard();
      }
    });
  }

  Future<bool> _prepareCaptureService() async {
    final nativeCaptureReady = await _nativeBridge.isScreenCaptureReady();

    if (!_captureReady || !nativeCaptureReady) {
      final granted = await _nativeBridge.requestScreenCapturePermission();
      if (!granted) {
        await _publishStatus('Screen capture permission denied');
        return false;
      }
      _captureReady = true;
    }

    await _nativeBridge.startOverlayService();
    await Future.delayed(const Duration(milliseconds: 120));
    return true;
  }

  Future<void> _setOverlayClickThrough(bool enabled) async {
    if (!_isOverlayShowing) {
      return;
    }

    try {
      await FlutterOverlayWindow.updateFlag(
        enabled ? OverlayFlag.clickThrough : OverlayFlag.defaultFlag,
      );
    } catch (_) {
      // Ignore runtime flag update failures on unsupported devices.
    }
  }

  Future<void> _showOverlay() async {
    final dpr = MediaQuery.of(context).devicePixelRatio;

    if (!await FlutterOverlayWindow.isPermissionGranted()) {
      setState(() => _status = 'Grant overlay permission first!');
      return;
    }

    final isCaptureReady = await _prepareCaptureService();
    if (!isCaptureReady) {
      return;
    }

    OverlayDataBridge.gridLeft = double.tryParse(_leftCtrl.text) ?? 70;
    OverlayDataBridge.gridTop = double.tryParse(_topCtrl.text) ?? 420;
    OverlayDataBridge.cellSize = double.tryParse(_cellSizeCtrl.text) ?? 120;
    OverlayDataBridge.originalBoard = _board.copy();
    OverlayDataBridge.solvedBoard = _solvedBoard.copy();
    OverlayDataBridge.status = _solvedBoard.filledCount > 0
        ? 'Solution ready'
        : 'No solution yet';

    const targetWidthDp = 400.0;
    const targetHeightDp = 250.0;
    final overlayWidthPx = (targetWidthDp * dpr).round();
    final overlayHeightPx = (targetHeightDp * dpr).round();
    final startX = (12 * dpr).roundToDouble();
    final startY = (90 * dpr).roundToDouble();

    await FlutterOverlayWindow.showOverlay(
      enableDrag: true,
      overlayTitle: "Sudoku Solver",
      overlayContent: "Scan board, then fill board",
      flag: OverlayFlag.defaultFlag,
      visibility: NotificationVisibility.visibilityPublic,
      positionGravity: PositionGravity.none,
      height: overlayHeightPx,
      width: overlayWidthPx,
      startPosition: OverlayPosition(startX, startY),
    );

    await FlutterOverlayWindow.shareData(
      jsonEncode(OverlayDataBridge.toJson()),
    );

    setState(() {
      _isOverlayShowing = true;
      _status = 'Overlay active! Drag & position over Sudoku grid.';
    });
  }

  Future<void> _hideOverlay() async {
    await FlutterOverlayWindow.closeOverlay();
    await _nativeBridge.stopOverlayService();
    setState(() {
      _isOverlayShowing = false;
      _status = 'Overlay hidden';
    });
  }

  Future<void> _fillSolvedBoard() async {
    if (_solvedBoard.isEmpty) {
      await _publishStatus('No solved board yet. Scan Board first.');
      return;
    }

    await _publishStatus('Filling board from solved scan...');
    final didFill = await _autoTapSolution(
      explicitRect: _lastSolvedBoardRect,
      imageWidth: _lastCaptureWidth,
      imageHeight: _lastCaptureHeight,
    );

    if (didFill) {
      await _publishStatus('Fill Board completed successfully.');
    } else {
      await _publishStatus('Fill Board failed or was blocked.');
    }
  }

  Future<bool> _autoTapSolution({
    List<double>? explicitRect,
    double? imageWidth,
    double? imageHeight,
  }) async {
    if (_solvedBoard.isEmpty) {
      setState(() => _status = 'No solution to fill! Solve a puzzle first.');
      return false;
    }

    setState(() {
      _isBusy = true;
      _status = 'Auto-tapping solution...';
    });

    try {
      await _setOverlayClickThrough(true);

      final accessibilityEnabled = await _nativeBridge
          .isAccessibilityServiceEnabled();
      if (!accessibilityEnabled) {
        await _publishStatus(
          'Accessibility service is OFF. Enable "Sudoku Solver" in Accessibility settings.',
        );
        return false;
      }

      final gridLeft = explicitRect != null
          ? explicitRect[0]
          : (double.tryParse(_leftCtrl.text) ?? 70);
      final gridTop = explicitRect != null
          ? explicitRect[1]
          : (double.tryParse(_topCtrl.text) ?? 420);
      final cellSize = explicitRect != null
          ? explicitRect[2]
          : (double.tryParse(_cellSizeCtrl.text) ?? 120);

      int filledCount = 0;

      for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
          if (_board.cells[row][col] != 0) continue;

          final value = _solvedBoard.cells[row][col];
          if (value == 0) continue;

          final cellX = (col * cellSize + gridLeft + cellSize / 2).toInt();
          final cellY = (row * cellSize + gridTop + cellSize / 2).toInt();

          final didTapCell = await _nativeBridge.performTap(cellX, cellY);
          if (!didTapCell) {
            await _publishStatus(
              'Tap blocked. Check Accessibility permission and keep overlay enabled.',
            );
            return false;
          }
          await Future.delayed(const Duration(milliseconds: 160));

          final numCoords = _resolveNumberCoords(
            value,
            boardRect: explicitRect,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
          );
          if (numCoords != null) {
            final didTapNumber = await _nativeBridge.performTap(
              numCoords[0],
              numCoords[1],
            );
            if (!didTapNumber) {
              await _publishStatus(
                'Number tap blocked. Check Accessibility permission.',
              );
              return false;
            }
            await Future.delayed(const Duration(milliseconds: 170));
          }

          filledCount++;
          if (filledCount == 1 || filledCount % 5 == 0) {
            setState(() => _status = 'Filling cell ($row,$col) = $value...');
          }
        }
      }

      setState(() => _status = 'Done! Filled $filledCount cells.');
      return true;
    } catch (e) {
      setState(() => _status = 'Auto-tap error: $e');
      return false;
    } finally {
      await _setOverlayClickThrough(false);
      setState(() => _isBusy = false);
    }
  }

  List<int>? _resolveNumberCoords(
    int value, {
    List<double>? boardRect,
    double? imageWidth,
    double? imageHeight,
  }) {
    final manual = _numberCoords[value];
    if (boardRect == null || boardRect.length < 3) {
      return manual;
    }
    if (imageWidth == null || imageHeight == null) {
      return manual;
    }

    final left = boardRect[0];
    final top = boardRect[1];
    final cellSize = boardRect[2];
    final boardSize = cellSize * 9;
    final boardRight = left + boardSize;
    final boardBottom = top + boardSize;

    final rightSpace = imageWidth - boardRight;
    final bottomSpace = imageHeight - boardBottom;

    // Most web Sudoku layouts use a vertical keypad on the right side.
    if (rightSpace > bottomSpace && rightSpace > cellSize * 0.8) {
      final x = (boardRight + rightSpace * 0.62).toInt();
      final y = (top + boardSize * (0.18 + ((value - 1).clamp(0, 8) * 0.11)))
          .toInt();
      return [x, y];
    }

    // Fallback for layouts with number buttons below the board.
    if (bottomSpace > cellSize * 0.6) {
      final x = (left + boardSize * (0.06 + ((value - 1).clamp(0, 8) * 0.11)))
          .toInt();
      final y = (boardBottom + bottomSpace * 0.68).toInt();
      return [x, y];
    }

    return manual;
  }

  // ==================== PERMISSIONS ====================

  Future<void> _requestOverlayPermission() async {
    var granted = await FlutterOverlayWindow.isPermissionGranted();
    if (!granted) {
      await FlutterOverlayWindow.requestPermission();
      granted = await FlutterOverlayWindow.isPermissionGranted();
    }
    setState(
      () => _status = granted
          ? 'Overlay permission granted!'
          : 'Please grant overlay permission',
    );
  }

  Future<void> _openAccessibilitySettings() async {
    await _nativeBridge.openAccessibilitySettings();
    setState(
      () => _status =
          'Enable "Sudoku Solver" in Accessibility settings for auto-tap',
    );
  }

  Future<void> _openDebugToolsMenu() async {
    await showModalBottomSheet<void>(
      context: context,
      builder: (sheetContext) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const ListTile(
                title: Text(
                  'Debug Tools',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                subtitle: Text('Pipeline checks'),
              ),
              ListTile(
                leading: const Icon(Icons.bug_report),
                title: const Text('Pipeline Test'),
                subtitle: const Text('Inspect OCR pipeline from camera/gallery'),
                onTap: () {
                  Navigator.pop(sheetContext);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const TestPipelineScreen(),
                    ),
                  );
                },
              ),
              const SizedBox(height: 8),
            ],
          ),
        );
      },
    );
  }

  // ==================== HELPERS ====================

  void _printBoard(String label, List<List<int>> board) {
    print('\n$label Board:');
    for (var row in board) {
      print(row.map((n) => n == 0 ? '.' : '$n').join(' '));
    }
  }

  void _clearBoard() {
    _board.clear();
    _solvedBoard.clear();
    _lastSolvedBoardRect = null;
    _previewImageBytes = null;
    _previewImageWidth = null;
    _previewImageHeight = null;
    OverlayDataBridge.clear();
    setState(() => _status = 'Board cleared');
  }

  // ==================== BUILD ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sudoku Solver'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.bug_report),
            tooltip: 'Debug Tools',
            onPressed: _openDebugToolsMenu,
          ),
        ],
      ),
      body: SafeArea(
        child: DefaultTabController(
          length: 2,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
                child: _buildStatusCard(),
              ),
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const TabBar(
                  tabs: [
                    Tab(icon: Icon(Icons.camera_alt), text: 'Camera'),
                    Tab(icon: Icon(Icons.layers), text: 'Overlay'),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: TabBarView(
                  children: [
                    _buildModePane([
                      _buildScanButtons(),
                      const SizedBox(height: 12),
                      _buildSolvedOverlayPreviewCard(),
                      const SizedBox(height: 12),
                      _buildCameraTipsCard(),
                    ]),
                    _buildModePane([
                      _buildOverlayControls(),
                      const SizedBox(height: 12),
                      _buildCalibrationCard(),
                      const SizedBox(height: 12),
                      _buildPermissionsCard(),
                      const SizedBox(height: 12),
                      _buildOverlayGuideCard(),
                    ]),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildModePane(List<Widget> children) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 16),
      children: children,
    );
  }

  Widget _buildStatusCard() {
    final isError =
        _status.toLowerCase().contains('error') ||
        _status.toLowerCase().contains('invalid');
    final isSuccess =
        _status.toLowerCase().contains('solved') ||
        _status.toLowerCase().contains('done');

    return Card(
      color: isError
          ? Colors.red.shade50
          : (isSuccess ? Colors.green.shade50 : Colors.blue.shade50),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(
              isError
                  ? Icons.error
                  : (isSuccess ? Icons.check_circle : Icons.info),
              color: isError
                  ? Colors.red
                  : (isSuccess ? Colors.green : Colors.blue),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(_status, style: const TextStyle(fontSize: 13)),
            ),
            if (_isBusy)
              const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildScanButtons() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Camera Mode',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isBusy || !_isModelLoaded
                        ? null
                        : _solveFromCamera,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Scan with Camera'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isBusy || !_isModelLoaded
                        ? null
                        : _solveFromGallery,
                    icon: const Icon(Icons.image),
                    label: const Text('Import from Gallery'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.purple,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraTipsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text(
              'Best Capture Tips',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 6),
            Text('Keep the Sudoku board fully visible in frame.'),
            Text('Avoid motion blur and strong reflections.'),
            Text('Use bright lighting for better OCR detection.'),
          ],
        ),
      ),
    );
  }

  Widget _buildSolvedOverlayPreviewCard() {
    if (_previewImageBytes == null ||
        _previewImageWidth == null ||
        _previewImageHeight == null ||
        _solvedBoard.isEmpty) {
      return const SizedBox.shrink();
    }

    final imageSize = Size(_previewImageWidth!, _previewImageHeight!);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Solved Overlay Preview',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
                TextButton.icon(
                  onPressed: _clearBoard,
                  icon: const Icon(Icons.clear, size: 16),
                  label: const Text('Clear'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            AspectRatio(
              aspectRatio: imageSize.width / imageSize.height,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(_previewImageBytes!, fit: BoxFit.contain),
                    IgnorePointer(
                      child: CustomPaint(
                        painter: _SolvedNumbersOverlayPainter(
                          originalBoard: _board.cells,
                          solvedBoard: _solvedBoard.cells,
                          boardRect: _lastSolvedBoardRect,
                          imageSize: imageSize,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCalibrationCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '📐 Grid Calibration',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _leftCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Left',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    keyboardType: TextInputType.number,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextField(
                    controller: _topCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Top',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    keyboardType: TextInputType.number,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextField(
                    controller: _cellSizeCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Cell',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    keyboardType: TextInputType.number,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildOverlayControls() {
    return Card(
      color: const Color(0xFF0D233F),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Overlay Mode',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              _isOverlayShowing
                  ? 'Overlay is active. Use Scan Board first, then Fill Board.'
                  : 'Start overlay to use floating controls over the Sudoku app.',
              style: TextStyle(color: Colors.lightBlue.shade100, fontSize: 12),
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _isOverlayShowing ? _hideOverlay : _showOverlay,
                icon: Icon(
                  _isOverlayShowing ? Icons.visibility_off : Icons.visibility,
                ),
                label: Text(
                  _isOverlayShowing ? 'Hide Overlay' : 'Show Overlay',
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isOverlayShowing
                      ? const Color(0xFFC62828)
                      : const Color(0xFF2E7D32),
                  foregroundColor: Colors.white,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isBusy || !_isModelLoaded
                        ? null
                        : () => _solveFromScreen(autoFill: false),
                    icon: const Icon(Icons.document_scanner),
                    label: const Text('Scan Board'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1565C0),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isBusy ? null : _fillSolvedBoard,
                    icon: const Icon(Icons.touch_app),
                    label: const Text('Fill Board'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFEF6C00),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              'Auto-fill needs Accessibility Service enabled.',
              style: TextStyle(fontSize: 12, color: Colors.white),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildOverlayGuideCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text('Overlay Flow', style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 6),
            Text('1) Enable Overlay and Accessibility permissions.'),
            Text('2) Show overlay and open the Sudoku app.'),
            Text('3) Tap "Scan Board" to capture and solve.'),
            Text('4) Tap "Fill Board" to enter only empty cells.'),
          ],
        ),
      ),
    );
  }

  Widget _buildPermissionsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '🔐 Permissions',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _requestOverlayPermission,
                    icon: const Icon(Icons.layers, size: 16),
                    label: const Text(
                      'Overlay',
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _openAccessibilitySettings,
                    icon: const Icon(Icons.accessibility, size: 16),
                    label: const Text(
                      'Accessibility',
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SolvedNumbersOverlayPainter extends CustomPainter {
  _SolvedNumbersOverlayPainter({
    required this.originalBoard,
    required this.solvedBoard,
    required this.boardRect,
    required this.imageSize,
  });

  final List<List<int>> originalBoard;
  final List<List<int>> solvedBoard;
  final List<double>? boardRect;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    if (boardRect == null || boardRect!.length < 3) {
      return;
    }
    if (imageSize.width <= 0 || imageSize.height <= 0) {
      return;
    }

    final fitted = applyBoxFit(BoxFit.contain, imageSize, size);
    final imageRect = Alignment.center.inscribe(
      fitted.destination,
      Offset.zero & size,
    );

    final scaleX = imageRect.width / imageSize.width;
    final scaleY = imageRect.height / imageSize.height;
    final scale = (scaleX + scaleY) / 2;

    final left = imageRect.left + boardRect![0] * scale;
    final top = imageRect.top + boardRect![1] * scale;
    final cellSize = boardRect![2] * scale;

    if (cellSize <= 0) {
      return;
    }

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (int row = 0; row < 9; row++) {
      for (int col = 0; col < 9; col++) {
        final original = originalBoard[row][col];
        final solved = solvedBoard[row][col];
        if (original != 0 || solved == 0) {
          continue;
        }

        final fontSize = (cellSize * 0.58).clamp(10.0, 44.0);
        textPainter.text = TextSpan(
          text: '$solved',
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.w700,
            color: const Color(0xFF2E7D32),
            shadows: const [
              Shadow(offset: Offset(1, 1), blurRadius: 2, color: Colors.white),
            ],
          ),
        );
        textPainter.layout();

        final dx = left + col * cellSize + (cellSize - textPainter.width) / 2;
        final dy = top + row * cellSize + (cellSize - textPainter.height) / 2;
        textPainter.paint(canvas, Offset(dx, dy));
      }
    }
  }

  @override
  bool shouldRepaint(covariant _SolvedNumbersOverlayPainter oldDelegate) {
    return oldDelegate.boardRect != boardRect ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.originalBoard != originalBoard ||
        oldDelegate.solvedBoard != solvedBoard;
  }
}

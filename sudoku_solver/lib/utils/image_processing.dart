import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as dart_img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

class TestProcessingResult {
  final dart_img.Image original;
  final dart_img.Image? thresholded; // For debugging full board
  final dart_img.Image? inverted;
  final dart_img.Image? board;
  final List<dart_img.Image> cellImages; // Extracted cells
  final String logs;

  TestProcessingResult({
    required this.original,
    this.thresholded,
    this.inverted,
    this.board,
    required this.cellImages,
    required this.logs,
  });
}

class ImageProcessor {
  static const int boardSize = 900;
  static const int inputSize = 48;
  static const int maxInputSide = 2200;
  static const int maxProcessingSide = 1800;

  // Helpers to convert cv.Mat to dart_img.Image for UI
  static dart_img.Image matToDartImage(cv.Mat mat) {
    var (_, bytes) = cv.imencode('.png', mat);
    return dart_img.decodeImage(bytes)!;
  }

  static List<cv.Point2f> orderPoints(cv.VecPoint pts) {
    var pList = pts.toList();
    if (pList.length != 4) return [];

    // sum
    var sortedBySum = List<cv.Point>.from(pList)
      ..sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    var topLeft = cv.Point2f(
      sortedBySum.first.x.toDouble(),
      sortedBySum.first.y.toDouble(),
    );
    var bottomRight = cv.Point2f(
      sortedBySum.last.x.toDouble(),
      sortedBySum.last.y.toDouble(),
    );

    // diff
    var sortedByDiff = List<cv.Point>.from(pList)
      ..sort((a, b) => (a.x - a.y).compareTo(b.x - b.y));
    var bottomLeft = cv.Point2f(
      sortedByDiff.first.x.toDouble(),
      sortedByDiff.first.y.toDouble(),
    );
    var topRight = cv.Point2f(
      sortedByDiff.last.x.toDouble(),
      sortedByDiff.last.y.toDouble(),
    );

    return [topLeft, topRight, bottomRight, bottomLeft];
  }

  static double _pointDistance(cv.Point2f a, cv.Point2f b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
  }

  static cv.Mat? getPerspective(cv.Mat img, cv.VecPoint location) {
    var ordered = orderPoints(location);
    if (ordered.isEmpty) return null;

    var src = cv.VecPoint2f.fromList(ordered);
    var dst = cv.VecPoint2f.fromList([
      cv.Point2f(0, 0),
      cv.Point2f(boardSize.toDouble(), 0),
      cv.Point2f(boardSize.toDouble(), boardSize.toDouble()),
      cv.Point2f(0, boardSize.toDouble()),
    ]);

    var matrix = cv.getPerspectiveTransform2f(src, dst);
    var result = cv.warpPerspective(img, matrix, (boardSize, boardSize));
    return result;
  }

  static TestProcessingResult? processImageForTest(Uint8List imageBytes) {
    StringBuffer logs = StringBuffer();
    void log(String msg) {
      logs.writeln(msg);
      print(msg);
    }

    // Fallback: decode with image package for Original UI view
    dart_img.Image? origImgForDart = dart_img.decodeImage(imageBytes);
    if (origImgForDart == null) return null;

    log('1. Fix Camera Rotation EXIF');
    origImgForDart = dart_img.bakeOrientation(origImgForDart);
    log(
      'Image fixed orientation: ${origImgForDart.width}x${origImgForDart.height}',
    );

    final dart_img.Image originalImageUi = dart_img.copyResize(
      origImgForDart,
      width: origImgForDart.width,
      height: origImgForDart.height,
    );

    // Decode with OpenCV
    var cleanBytes = dart_img.encodePng(origImgForDart);
    var mat = cv.imdecode(Uint8List.fromList(cleanBytes), cv.IMREAD_COLOR);
    if (mat.isEmpty) {
      log('OpenCV decode failed.');
      return null;
    }

    log('2. Check size: ${mat.cols}x${mat.rows}');
    int h = mat.rows;
    int w = mat.cols;
    double scale = 1.0;
    if (max(h, w) > maxProcessingSide) {
      scale = maxProcessingSide / max(h, w);
      log(
        'Downscaled to: ${(w * scale).toInt()}x${(h * scale).toInt()} (max $maxProcessingSide limit)',
      );
      var newMat = cv.resize(mat, (
        (w * scale).toInt(),
        (h * scale).toInt(),
      ), interpolation: cv.INTER_AREA);
      mat.dispose();
      mat = newMat;
    }

    log('3. Convert to grayscale and process');
    var gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);
    var blur = cv.gaussianBlur(gray, (9, 9), 0);
    var thresh = cv.adaptiveThreshold(
      blur,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      11,
      2,
    );

    log('4. Find the Sudoku board using contour detection...');
    var inverted = cv.bitwiseNOT(thresh);

    var (contours, _) = cv.findContours(
      inverted,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    var contourList = contours.toList();
    log('Found ${contourList.length} external contours.');

    // Sort contours by area
    contourList.sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));

    cv.VecPoint? location;
    for (var contour in contourList.take(15)) {
      double peri = cv.arcLength(contour, true);
      var approx = cv.approxPolyDP(contour, 0.02 * peri, true);
      if (approx.length == 4) {
        location = approx;
        log('Found board contour');
        break;
      }
    }

    if (location == null) {
      log(
        'Warning: Advanced thresholding failed, falling back to Canny edge detection.',
      );
      var bFilter = cv.bilateralFilter(gray, 13, 20, 20);
      var edged = cv.canny(bFilter, 30, 180);
      var (edgeContours, _) = cv.findContours(
        edged,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
      );
      var edgeList = edgeContours.toList();
      edgeList.sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));
      for (var contour in edgeList.take(15)) {
        var approx = cv.approxPolyDP(contour, 15, true);
        if (approx.length == 4) {
          location = approx;
          log('Found board contour via Canny');
          break;
        }
      }
    }

    cv.Mat? boardResultUI;
    cv.Mat? boardResult;
    if (location != null) {
      boardResultUI = getPerspective(mat, location);
      boardResult = getPerspective(gray, location);
      log('Successfully applied perspective transform');
    } else {
      log('Could not find board contour! Fallback to center crop.');
      final size = min(gray.cols, gray.rows);
      final startX = (gray.cols - size) ~/ 2;
      final startY = (gray.rows - size) ~/ 2;
      var roi = cv.Rect(startX, startY, size, size);
      var croppedColor = mat.region(roi);
      boardResultUI = cv.resize(croppedColor, (boardSize, boardSize));
      var cropped = gray.region(roi);
      boardResult = cv.resize(cropped, (boardSize, boardSize));
    }

    log('5. Splitting board cells...');
    List<dart_img.Image> cellImages = [];
    final cellSize = boardSize ~/ 9;

    for (int row = 0; row < 9; row++) {
      for (int col = 0; col < 9; col++) {
        final cellX = col * cellSize;
        final cellY = row * cellSize;
        var cellRoi = cv.Rect(cellX, cellY, cellSize, cellSize);
        var cell = boardResult!.region(cellRoi);

        final ch = cell.rows;
        final cw = cell.cols;
        final padH = (ch * 0.15).toInt();
        final padW = (cw * 0.15).toInt();
        var croppedCell = cell.region(
          cv.Rect(padW, padH, cw - 2 * padW, ch - 2 * padH),
        );

        var blurredCell = cv.gaussianBlur(croppedCell, (5, 5), 0);
        var threshCell = cv.adaptiveThreshold(
          blurredCell,
          255,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY_INV,
          15,
          15,
        );

        var kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
        var cleaned = cv.morphologyEx(threshCell, cv.MORPH_OPEN, kernel);

        cellImages.add(matToDartImage(cleaned));
      }
    }

    return TestProcessingResult(
      original: originalImageUi,
      thresholded: matToDartImage(thresh),
      inverted: matToDartImage(inverted),
      board: boardResultUI != null ? matToDartImage(boardResultUI) : null,
      cellImages: cellImages,
      logs: logs.toString(),
    );
  }

  /// Process image and extract 81 cell inputs for OCR
  static Map<String, dynamic>? processImage(
    Uint8List imageBytes, {
    bool alreadyNormalized = false,
  }) {
    Uint8List normalizedBytes;
    final decoded = dart_img.decodeImage(imageBytes);
    if (decoded == null) {
      if (!alreadyNormalized) {
        return null;
      }
      normalizedBytes = imageBytes;
    } else {
      var oriented = dart_img.bakeOrientation(decoded);
      final longSide = max(oriented.width, oriented.height);
      if (longSide > maxInputSide) {
        final scale = maxInputSide / longSide;
        oriented = dart_img.copyResize(
          oriented,
          width: max(1, (oriented.width * scale).round()),
          height: max(1, (oriented.height * scale).round()),
        );
      }

      normalizedBytes = Uint8List.fromList(dart_img.encodePng(oriented));
    }

    var mat = cv.imdecode(normalizedBytes, cv.IMREAD_COLOR);
    if (mat.isEmpty) return null;

    int h = mat.rows;
    int w = mat.cols;
    double usedScale = 1.0;
    if (max(h, w) > maxProcessingSide) {
      usedScale = maxProcessingSide / max(h, w);
      var resized = cv.resize(mat, (
        (w * usedScale).toInt(),
        (h * usedScale).toInt(),
      ), interpolation: cv.INTER_AREA);
      mat.dispose();
      mat = resized;
    }

    var gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);
    var blur = cv.gaussianBlur(gray, (9, 9), 0);
    var thresh = cv.adaptiveThreshold(
      blur,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      11,
      2,
    );
    var inverted = cv.bitwiseNOT(thresh);

    var (contours, _) = cv.findContours(
      inverted,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    var contourList = contours.toList();
    contourList.sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));

    cv.VecPoint? location;
    for (var contour in contourList.take(15)) {
      double peri = cv.arcLength(contour, true);
      var approx = cv.approxPolyDP(contour, 0.02 * peri, true);
      if (approx.length == 4) {
        location = approx;
        break;
      }
    }

    if (location == null) {
      var bFilter = cv.bilateralFilter(gray, 13, 20, 20);
      var edged = cv.canny(bFilter, 30, 180);
      var (edgeContours, _) = cv.findContours(
        edged,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
      );
      var edgeList = edgeContours.toList();
      edgeList.sort((a, b) => cv.contourArea(b).compareTo(cv.contourArea(a)));
      for (var contour in edgeList.take(15)) {
        var approx = cv.approxPolyDP(contour, 15, true);
        if (approx.length == 4) {
          location = approx;
          break;
        }
      }
    }

    cv.Mat? boardResult;
    double boardLeft = 0;
    double boardTop = 0;
    double boardWidth = 0;
    double boardHeight = 0;
    if (location != null) {
      var ordered = orderPoints(location);
      if (ordered.isNotEmpty) {
        final xs = ordered.map((p) => p.x).toList();
        final ys = ordered.map((p) => p.y).toList();

        boardLeft = xs.reduce(min);
        boardTop = ys.reduce(min);

        final topWidth = _pointDistance(ordered[0], ordered[1]);
        final bottomWidth = _pointDistance(ordered[3], ordered[2]);
        final leftHeight = _pointDistance(ordered[0], ordered[3]);
        final rightHeight = _pointDistance(ordered[1], ordered[2]);

        boardWidth = max(topWidth, bottomWidth);
        boardHeight = max(leftHeight, rightHeight);
      }
      boardResult = getPerspective(gray, location);
    } else {
      final size = min(gray.cols, gray.rows);
      final startX = (gray.cols - size) ~/ 2;
      final startY = (gray.rows - size) ~/ 2;
      boardLeft = startX.toDouble();
      boardTop = startY.toDouble();
      boardWidth = size.toDouble();
      boardHeight = size.toDouble();
      var roi = cv.Rect(startX, startY, size, size);
      var cropped = gray.region(roi);
      boardResult = cv.resize(cropped, (boardSize, boardSize));
    }

    boardLeft = max(0.0, boardLeft / usedScale);
    boardTop = max(0.0, boardTop / usedScale);
    boardWidth = boardWidth / usedScale;
    boardHeight = boardHeight / usedScale;

    final normalizedBoardSize = boardHeight > 0
        ? (boardWidth + boardHeight) / 2.0
        : boardWidth;
    final calcCellSize = (normalizedBoardSize / 9.0)
        .clamp(8.0, 400.0)
        .toDouble();

    if (boardResult == null) return null;

    final cellSize = boardSize ~/ 9;
    final batchInput = <List<List<List<double>>>?>[];

    for (int row = 0; row < 9; row++) {
      for (int col = 0; col < 9; col++) {
        final cellX = col * cellSize;
        final cellY = row * cellSize;
        var cellRoi = cv.Rect(cellX, cellY, cellSize, cellSize);
        var cell = boardResult.region(cellRoi);

        final ch = cell.rows;
        final cw = cell.cols;
        final padH = (ch * 0.15).toInt();
        final padW = (cw * 0.15).toInt();
        var croppedCell = cell.region(
          cv.Rect(padW, padH, cw - 2 * padW, ch - 2 * padH),
        );

        var blurredCell = cv.gaussianBlur(croppedCell, (5, 5), 0);
        var threshCell = cv.adaptiveThreshold(
          blurredCell,
          255,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY_INV,
          15,
          15,
        );

        var kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
        var cleaned = cv.morphologyEx(threshCell, cv.MORPH_OPEN, kernel);

        var (cellCnts, _) = cv.findContours(
          cleaned,
          cv.RETR_EXTERNAL,
          cv.CHAIN_APPROX_SIMPLE,
        );
        var cellCntList = cellCnts.toList();

        bool isEmpty = true;
        cv.Rect? digitRect;

        if (cellCntList.isNotEmpty) {
          List<cv.VecPoint> validCnts = [];
          for (var c in cellCntList) {
            var rect = cv.boundingRect(c);
            var area = cv.contourArea(c);

            bool validHeight =
                croppedCell.rows * 0.85 > rect.height &&
                rect.height > croppedCell.rows * 0.20;
            bool validWidth =
                croppedCell.cols * 0.85 > rect.width &&
                rect.width > croppedCell.cols * 0.05;
            bool validArea = area > 10;
            bool touchingEdge =
                (rect.x == 0) ||
                (rect.y == 0) ||
                (rect.x + rect.width >= croppedCell.cols - 1) ||
                (rect.y + rect.height >= croppedCell.rows - 1);

            if (validHeight && validWidth && validArea && !touchingEdge) {
              validCnts.add(c);
            }
          }

          if (validCnts.isNotEmpty) {
            isEmpty = false;
            int minX = 9999, minY = 9999, maxX = 0, maxY = 0;
            for (var c in validCnts) {
              var r = cv.boundingRect(c);
              if (r.x < minX) minX = r.x;
              if (r.y < minY) minY = r.y;
              if (r.x + r.width > maxX) maxX = r.x + r.width;
              if (r.y + r.height > maxY) maxY = r.y + r.height;
            }
            digitRect = cv.Rect(minX, minY, maxX - minX, maxY - minY);
          }
        }

        List<List<List<double>>>? output;

        if (isEmpty || digitRect == null) {
          output = null;
        } else {
          output = List.generate(
            inputSize,
            (y) => List.generate(inputSize, (x) => [1.0]),
          );

          var digitMask = cleaned.region(digitRect);
          var digitImg = cv.bitwiseNOT(digitMask);

          final margin = (inputSize * 0.2).toInt();
          final tgtSize = inputSize - 2 * margin;

          double scaleBox = 1.0;
          if (digitRect.width > 0 && digitRect.height > 0) {
            scaleBox = min(
              tgtSize / digitRect.width,
              tgtSize / digitRect.height,
            );
          }
          final newW = max(1, (digitRect.width * scaleBox).toInt());
          final newH = max(1, (digitRect.height * scaleBox).toInt());

          var resized = cv.resize(digitImg, (
            newW,
            newH,
          ), interpolation: cv.INTER_AREA);

          final startX = ((inputSize - newW) ~/ 2).clamp(0, inputSize - 1);
          final startY = ((inputSize - newH) ~/ 2).clamp(0, inputSize - 1);

          final bytes = resized.data;

          for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
              if (startY + y < inputSize && startX + x < inputSize) {
                int val = bytes[y * newW + x];
                output[startY + y][startX + x][0] = val / 255.0;
              }
            }
          }
        }
        batchInput.add(output);
      }
    }

    return {
      'batch': batchInput,
      'rect': [boardLeft, boardTop, calcCellSize],
    };
  }
}

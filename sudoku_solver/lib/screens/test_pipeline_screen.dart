import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;

import '../utils/image_processing2.dart';
import '../services/ocr_engine.dart'; // Using correct import

class TestPipelineScreen extends StatefulWidget {
  const TestPipelineScreen({super.key});

  @override
  State<TestPipelineScreen> createState() => _TestPipelineScreenState();
}

class _TestPipelineScreenState extends State<TestPipelineScreen> {
  bool _isProcessing = false;
  TestProcessingResult? _result;
  SudokuOcrEngine? _ocrEngine;
  List<int>? _predictions;
  String _lastSource = 'None';
  String? _boardDebugText;

  @override
  void initState() {
    super.initState();
    _ocrEngine = SudokuOcrEngine();
    _ocrEngine!.loadModel();
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(
      source: source,
      imageQuality: 96,
      maxWidth: 2200,
      maxHeight: 2200,
    );
    if (pickedFile == null) return;

    setState(() {
      _isProcessing = true;
      _result = null;
      _predictions = null;
      _boardDebugText = null;
      _lastSource = source == ImageSource.camera ? 'Camera' : 'Gallery';
    });

    final bytes = await File(pickedFile.path).readAsBytes();

    // Also run OCR for predictions
    List<List<int>>? board;
    try {
      final res = await _ocrEngine!.processImage(bytes);
      board = res?['board'] as List<List<int>>?;
      if (board != null) {
        _boardDebugText = board
            .map((row) => row.map((n) => n == 0 ? '.' : '$n').join(' '))
            .join('\n');
      }
    } catch (e) {
      print('OCR Error: $e');
    }

    final result = ImageProcessor.processImageForTest(bytes);

    setState(() {
      _result = result;
      _isProcessing = false;
      if (board != null) {
        _predictions = board.expand((row) => row).toList();
      }
    });
  }

  Widget _buildImage(String title, img.Image? image) {
    if (image == null) return const SizedBox();
    return Column(
      children: [
        Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
        ),
        const SizedBox(height: 8),
        Image.memory(
          Uint8List.fromList(img.encodePng(image)),
          height: 300,
          fit: BoxFit.contain,
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildCellsGrid() {
    if (_result == null || _result!.cellImages.isEmpty) return const SizedBox();

    return Column(
      children: [
        const Text(
          'Extracted Cells (Cleaned)',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
        ),
        const SizedBox(height: 8),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 9,
            crossAxisSpacing: 2,
            mainAxisSpacing: 2,
          ),
          itemCount: 81,
          itemBuilder: (context, index) {
            final cellImg = _result!.cellImages[index];
            final bytes = Uint8List.fromList(img.encodePng(cellImg));
            final pred = _predictions?[index] ?? 0;
            return Container(
              color: Colors.black,
              padding: const EdgeInsets.all(1),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.memory(bytes, fit: BoxFit.fill),
                  if (pred != 0)
                    Center(
                      child: Text(
                        pred.toString(),
                        style: const TextStyle(
                          color: Colors.red,
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                    ),
                ],
              ),
            );
          },
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Pipeline Test')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing
                        ? null
                        : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Use Camera'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing
                        ? null
                        : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.image),
                    label: const Text('Use Gallery'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text('Source: $_lastSource'),
            const SizedBox(height: 8),
            if (_isProcessing) const CircularProgressIndicator(),
            if (_result != null) ...[
              if (_boardDebugText != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  color: Colors.black87,
                  child: Text(
                    _boardDebugText!,
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: Colors.white,
                    ),
                  ),
                ),
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(8),
                color: Colors.grey[200],
                child: Text(
                  _result!.logs,
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
              ),
              const Divider(),
              _buildImage('Original Image', _result!.original),
              _buildImage('Thresholded', _result!.thresholded),
              _buildImage('Board Cropped', _result!.board),
              _buildCellsGrid(),
            ],
          ],
        ),
      ),
    );
  }
}

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../utils/image_processing2.dart';

/// TFLite OCR Engine - Matches main.py exactly
class SudokuOcrEngine {
  Interpreter? _interpreter;
  static const int inputSize = 48;
  static const double minDigitConfidence = 0.58;

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    try {
      final modelData = await rootBundle.load('assets/model-OCR.tflite');
      final buffer = modelData.buffer;
      _interpreter = Interpreter.fromBuffer(buffer.asUint8List());

      // Print model info for debugging
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      print('TFLite model loaded!');
      print('Input shape: $inputShape');
      print('Output shape: $outputShape');
    } catch (e) {
      print('TFLite load error: $e');
      rethrow;
    }
  }

  /// Process image and return detected board
  Future<Map<String, dynamic>?> processImage(
    Uint8List imageBytes, {
    bool alreadyNormalized = false,
  }) async {
    if (_interpreter == null) {
      print('Interpreter is null');
      return null;
    }

    try {
      // Run heavy image processing in background isolate.
      final processResult = await compute(_processImageInIsolate, {
        'bytes': imageBytes,
        'alreadyNormalized': alreadyNormalized,
      });
      if (processResult == null) return null;
      final batchInput =
          processResult['batch'] as List<List<List<List<double>>>?>;
      final boardRect = processResult['rect'] as List<double>;

      final result = List.generate(9, (_) => List.filled(9, 0));
      final confidence = List.generate(9, (_) => List.filled(9, 0.0));
      int lowConfidenceDrops = 0;

      // Run prediction for all 81 cells
      for (int i = 0; i < 81; i++) {
        final row = i ~/ 9;
        final col = i % 9;

        // Validated empty by ImageProcessor
        if (batchInput[i] == null) {
          result[row][col] = 0;
          continue;
        }

        // Input shape: [1, 48, 48, 1]
        final input = [batchInput[i]!];

        // Output shape: [1, 10] for classes 0-9
        final output = List.generate(1, (_) => List.filled(10, 0.0));

        _interpreter!.run(input, output);

        final predictions = output[0];
        int maxIdx = 0;
        double maxVal = predictions[0];

        for (int j = 1; j < 10; j++) {
          if (predictions[j] > maxVal) {
            maxVal = predictions[j];
            maxIdx = j;
          }
        }

        confidence[row][col] = maxVal;

        // Camera captures can produce uncertain OCR digits that make the
        // puzzle unsatisfiable. Keep only confident predictions.
        if (maxIdx == 0 || maxVal < minDigitConfidence) {
          if (maxIdx != 0 && maxVal < minDigitConfidence) {
            lowConfidenceDrops++;
          }
          result[row][col] = 0;
          continue;
        }

        result[row][col] = maxIdx;
      }

      return {
        'board': result,
        'rect': boardRect,
        'confidence': confidence,
        'lowConfidenceDrops': lowConfidenceDrops,
      };
    } catch (e) {
      print('Process error: $e');
      return null;
    }
  }

  void dispose() {
    _interpreter?.close();
  }
}

/// Isolate function for image processing
Map<String, dynamic>? _processImageInIsolate(Map<String, dynamic> payload) {
  final imageBytes = payload['bytes'] as Uint8List;
  final alreadyNormalized = payload['alreadyNormalized'] as bool? ?? false;
  return ImageProcessor.processImage(
    imageBytes,
    alreadyNormalized: alreadyNormalized,
  );
}

import 'package:flutter/services.dart';

/// Native bridge for Accessibility Service taps and screen capture
class NativeBridge {
  static const MethodChannel _channel = MethodChannel('sudoku_solver/native');

  Future<bool> isScreenCaptureReady() async {
    try {
      final bool? result = await _channel.invokeMethod('isScreenCaptureReady');
      return result ?? false;
    } catch (e) {
      print('Capture readiness error: $e');
      return false;
    }
  }

  Future<bool> requestScreenCapturePermission() async {
    try {
      final bool? result = await _channel.invokeMethod(
        'requestScreenCapturePermission',
      );
      return result ?? false;
    } catch (e) {
      print('Capture permission error: $e');
      return false;
    }
  }

  Future<void> startOverlayService() async {
    try {
      await _channel.invokeMethod('startOverlayService');
    } catch (e) {
      print('Start overlay service error: $e');
    }
  }

  Future<void> stopOverlayService() async {
    try {
      await _channel.invokeMethod('stopOverlayService');
    } catch (e) {
      print('Stop overlay service error: $e');
    }
  }

  Future<bool> performTap(int x, int y) async {
    try {
      final bool? result = await _channel.invokeMethod('performTap', {
        'x': x,
        'y': y,
      });
      return result ?? false;
    } catch (e) {
      print('Tap error: $e');
      return false;
    }
  }

  Future<bool> isAccessibilityServiceEnabled() async {
    try {
      final bool? result = await _channel.invokeMethod(
        'isAccessibilityServiceEnabled',
      );
      return result ?? false;
    } catch (e) {
      print('Accessibility status error: $e');
      return false;
    }
  }

  Future<String?> captureScreen() async {
    try {
      final String? path = await _channel.invokeMethod('captureScreen');
      return path;
    } catch (e) {
      print('Capture screen error: $e');
      return null;
    }
  }

  Future<void> openAccessibilitySettings() async {
    try {
      await _channel.invokeMethod('openAccessibilitySettings');
    } catch (e) {
      print('Settings error: $e');
    }
  }
}

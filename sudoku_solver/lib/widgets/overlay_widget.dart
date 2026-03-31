import 'dart:async';
import 'dart:convert';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_overlay_window/flutter_overlay_window.dart';

/// Overlay widget that shows on top of other apps for scanning and displaying status logs
class SudokuOverlayWidget extends StatefulWidget {
  const SudokuOverlayWidget({super.key});

  @override
  State<SudokuOverlayWidget> createState() => _SudokuOverlayWidgetState();
}

class _SudokuOverlayWidgetState extends State<SudokuOverlayWidget>
    with SingleTickerProviderStateMixin {
  String _status = 'Ready to scan';
  StreamSubscription<dynamic>? _overlaySub;
  Timer? _idleFadeTimer;
  Timer? _clickThroughResetTimer;
  bool _isPanelActive = true;
  bool _isClickThrough = false;
  late final AnimationController _pulseController;

  static const Duration _idleFadeDelay = Duration(milliseconds: 1800);

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);

    _markInteracting();

    _overlaySub = FlutterOverlayWindow.overlayListener.listen((data) {
      if (data == null || data.toString().isEmpty) {
        return;
      }

      _markInteracting();

      try {
        final json = jsonDecode(data.toString()) as Map<String, dynamic>;
        setState(() {
          _status = json['status']?.toString() ?? 'Updated';
        });
        if (_isClickThrough && _isTerminalStatus(_status)) {
          _disableClickThrough();
        }
      } catch (_) {
        setState(() {
          _status = data.toString();
        });
        if (_isClickThrough && _isTerminalStatus(_status)) {
          _disableClickThrough();
        }
      }
    });
  }

  @override
  void dispose() {
    _overlaySub?.cancel();
    _idleFadeTimer?.cancel();
    _clickThroughResetTimer?.cancel();
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _setOverlayFlag(OverlayFlag flag) async {
    try {
      await FlutterOverlayWindow.updateFlag(flag);
    } catch (_) {
      // Some devices/plugin states may reject runtime flag updates.
    }
  }

  void _enableClickThroughTemporarily({
    Duration duration = const Duration(seconds: 30),
  }) {
    if (!_isClickThrough && mounted) {
      setState(() => _isClickThrough = true);
    }
    _setOverlayFlag(OverlayFlag.clickThrough);
    _clickThroughResetTimer?.cancel();
    _clickThroughResetTimer = Timer(duration, () {
      if (!mounted) {
        return;
      }
      _disableClickThrough();
    });
  }

  void _disableClickThrough() {
    _clickThroughResetTimer?.cancel();
    if (_isClickThrough && mounted) {
      setState(() => _isClickThrough = false);
    }
    _setOverlayFlag(OverlayFlag.defaultFlag);
  }

  bool _isTerminalStatus(String message) {
    final status = message.toLowerCase();
    return status.contains('done') ||
        status.contains('success') ||
        status.contains('error') ||
        status.contains('failed') ||
        status.contains('invalid board') ||
        status.contains('no solution') ||
        status.contains('capture failed');
  }

  void _markInteracting() {
    if (!_isPanelActive && mounted) {
      setState(() => _isPanelActive = true);
    }

    _idleFadeTimer?.cancel();
    _idleFadeTimer = Timer(_idleFadeDelay, () {
      if (!mounted) {
        return;
      }
      setState(() => _isPanelActive = false);
    });
  }

  bool get _isWorking {
    final status = _status.toLowerCase();
    return status.contains('capturing') ||
        status.contains('processing') ||
        status.contains('auto-tap') ||
        status.contains('filling') ||
        status.contains('requesting') ||
        status.contains('solving');
  }

  bool get _isSuccess {
    final status = _status.toLowerCase();
    return status.contains('done') || status.contains('success');
  }

  bool get _isError {
    final status = _status.toLowerCase();
    return status.contains('error') || status.contains('failed');
  }

  Color get _accentColor {
    if (_isError) return const Color(0xFFFF6B6B);
    if (_isSuccess) return const Color(0xFF4CD964);
    if (_isWorking) return const Color(0xFF58A6FF);
    return const Color(0xFF6EA8FE);
  }

  IconData get _statusIcon {
    if (_isError) return Icons.error_outline;
    if (_isSuccess) return Icons.check_circle_outline;
    if (_isWorking) return Icons.autorenew;
    return Icons.info_outline;
  }

  Future<void> _sendOverlayAction(String action) async {
    final payloadMap = {
      'type': 'overlay_action',
      'action': action,
      'ts': DateTime.now().millisecondsSinceEpoch,
    };
    final payloadJson = jsonEncode(payloadMap);

    // Send multiple payload shapes because some devices/plugins behave differently
    // across app and overlay engines.
    for (final payload in [action, payloadMap, payloadJson]) {
      try {
        await FlutterOverlayWindow.shareData(payload);
      } catch (_) {
        // Continue trying other payload formats.
      }
      await Future.delayed(const Duration(milliseconds: 45));
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = _accentColor;
    final panelOpacity = _isPanelActive ? 1.0 : 0.62;

    return Material(
      color: Colors.transparent,
      child: Align(
        alignment: Alignment.topCenter,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
          child: LayoutBuilder(
            builder: (context, constraints) {
              final viewportWidth = constraints.maxWidth.isFinite
                  ? constraints.maxWidth
                  : 460.0;
              final viewportHeight = constraints.maxHeight.isFinite
                  ? constraints.maxHeight
                  : 620.0;

              final estimatedMaxWidth = viewportWidth - 4;
              final safeMaxWidth = estimatedMaxWidth <= 0
                  ? 220.0
                  : estimatedMaxWidth > 560.0
                  ? 560.0
                  : estimatedMaxWidth;
              final estimatedMaxHeight = viewportHeight - 24;
              final safeMaxHeight = estimatedMaxHeight <= 0
                  ? 220.0
                  : estimatedMaxHeight > 320.0
                  ? 320.0
                  : estimatedMaxHeight;
              final safeMinWidth = safeMaxWidth < 360 ? safeMaxWidth : 360.0;

              return ConstrainedBox(
                constraints: BoxConstraints(
                  minWidth: safeMinWidth,
                  maxWidth: safeMaxWidth,
                  maxHeight: safeMaxHeight,
                ),
                child: Listener(
                  behavior: HitTestBehavior.translucent,
                  onPointerDown: (_) => _markInteracting(),
                  onPointerMove: (_) => _markInteracting(),
                  onPointerUp: (_) => _markInteracting(),
                  child: AnimatedOpacity(
                    duration: const Duration(milliseconds: 260),
                    curve: Curves.easeOutCubic,
                    opacity: panelOpacity,
                    child: AnimatedBuilder(
                      animation: _pulseController,
                      builder: (context, child) {
                        final glow = 0.22 + (_pulseController.value * 0.25);

                        return AnimatedContainer(
                          duration: const Duration(milliseconds: 260),
                          curve: Curves.easeOutCubic,
                          margin: const EdgeInsets.only(top: 8),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(18),
                            gradient: const LinearGradient(
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                              colors: [Color(0xF0182538), Color(0xF0223048)],
                            ),
                            border: Border.all(
                              color: accent.withValues(alpha: glow),
                              width: 2,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: accent.withValues(alpha: glow * 0.6),
                                blurRadius: 18,
                                spreadRadius: 1,
                                offset: const Offset(0, 6),
                              ),
                              const BoxShadow(
                                color: Color(0x55000000),
                                blurRadius: 12,
                                offset: Offset(0, 8),
                              ),
                            ],
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: BackdropFilter(
                              filter: ImageFilter.blur(sigmaX: 4, sigmaY: 4),
                              child: Container(
                                padding: const EdgeInsets.fromLTRB(
                                  12,
                                  10,
                                  12,
                                  10,
                                ),
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Row(
                                      children: [
                                        Container(
                                          width: 22,
                                          height: 22,
                                          decoration: BoxDecoration(
                                            color: accent.withValues(
                                              alpha: 0.22,
                                            ),
                                            borderRadius: BorderRadius.circular(
                                              999,
                                            ),
                                          ),
                                          child: Icon(
                                            _statusIcon,
                                            size: 14,
                                            color: accent,
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        Expanded(
                                          child: Text(
                                            _status,
                                            key: ValueKey(_status),
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontSize: 13,
                                              fontWeight: FontWeight.w700,
                                              letterSpacing: 0.1,
                                            ),
                                            maxLines: 3,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                        ),
                                        const SizedBox(width: 6),
                                        IconButton(
                                          icon: const Icon(
                                            Icons.close,
                                            color: Colors.white,
                                            size: 20,
                                          ),
                                          onPressed: () {
                                            _markInteracting();
                                            FlutterOverlayWindow.closeOverlay();
                                          },
                                          padding: EdgeInsets.zero,
                                          constraints: const BoxConstraints(
                                            minWidth: 28,
                                            minHeight: 28,
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 8),
                                    if (_isWorking)
                                      ClipRRect(
                                        borderRadius: BorderRadius.circular(
                                          999,
                                        ),
                                        child: LinearProgressIndicator(
                                          minHeight: 4,
                                          color: accent,
                                          backgroundColor: Colors.white24,
                                        ),
                                      ),
                                    const SizedBox(height: 8),
                                    Row(
                                      children: [
                                        Expanded(
                                          child: ElevatedButton.icon(
                                            onPressed: () async {
                                              _markInteracting();
                                              setState(() {
                                                _status =
                                                    'Capturing and solving board...';
                                              });
                                              await _sendOverlayAction(
                                                'SCAN_BOARD_REQUEST',
                                              );
                                              _enableClickThroughTemporarily(
                                                duration: const Duration(
                                                  seconds: 8,
                                                ),
                                              );
                                            },
                                            icon: const Icon(
                                              Icons.document_scanner,
                                              size: 16,
                                            ),
                                            label: const Text('Scan Board'),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: const Color(
                                                0xFF1F7AE0,
                                              ),
                                              foregroundColor: Colors.white,
                                            ),
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        Expanded(
                                          child: ElevatedButton.icon(
                                            onPressed: () async {
                                              _markInteracting();
                                              setState(() {
                                                _status =
                                                    'Filling board with solved numbers...';
                                              });
                                              await _sendOverlayAction(
                                                'FILL_BOARD_REQUEST',
                                              );
                                              _enableClickThroughTemporarily();
                                            },
                                            icon: const Icon(
                                              Icons.touch_app,
                                              size: 16,
                                            ),
                                            label: const Text('Fill Board'),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: const Color(
                                                0xFFEF6C00,
                                              ),
                                              foregroundColor: Colors.white,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 6),
                                    const Text(
                                      'Use Scan Board first, then Fill Board.',
                                      style: TextStyle(
                                        color: Colors.white70,
                                        fontSize: 11,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}

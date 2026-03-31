import 'package:flutter/material.dart';
import '../models/sudoku_board.dart';

/// Widget to display Sudoku board preview with detected and solved cells
class BoardPreview extends StatelessWidget {
  final SudokuBoard board;
  final SudokuBoard solvedBoard;
  final VoidCallback? onClear;

  const BoardPreview({
    super.key,
    required this.board,
    required this.solvedBoard,
    this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final detectedCount = board.filledCount;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Board Preview:',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    Text(
                      'Detected: $detectedCount/81 cells',
                      style: TextStyle(
                        fontSize: 11,
                        color: detectedCount < 10
                            ? Colors.red
                            : (detectedCount < 17 ? Colors.orange : Colors.green),
                      ),
                    ),
                  ],
                ),
                TextButton.icon(
                  onPressed: onClear,
                  icon: const Icon(Icons.clear, size: 16),
                  label: const Text('Clear'),
                ),
              ],
            ),
            const SizedBox(height: 4),
            // Legend
            Row(
              children: [
                _buildLegendItem(Colors.blue.shade50, 'Detected'),
                const SizedBox(width: 12),
                _buildLegendItem(Colors.green.shade100, 'Solved'),
              ],
            ),
            const SizedBox(height: 8),
            AspectRatio(
              aspectRatio: 1,
              child: Container(
                decoration: BoxDecoration(
                    border: Border.all(width: 2, color: Colors.black)),
                child: GridView.builder(
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: 81,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 9),
                  itemBuilder: (context, index) {
                    final row = index ~/ 9;
                    final col = index % 9;
                    final original = board.cells[row][col];
                    final solved = solvedBoard.cells[row][col];
                    final isDetected = original != 0;
                    final isFilled = original == 0 && solved != 0;

                    return Container(
                      decoration: BoxDecoration(
                        border: Border(
                          right: BorderSide(
                              width: col % 3 == 2 && col != 8 ? 2 : 0.5),
                          bottom: BorderSide(
                              width: row % 3 == 2 && row != 8 ? 2 : 0.5),
                        ),
                        color: isDetected
                            ? Colors.blue.shade50
                            : (isFilled ? Colors.green.shade100 : null),
                      ),
                      child: Center(
                        child: Text(
                          isDetected
                              ? '$original'
                              : (solved != 0 ? '$solved' : ''),
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: isDetected
                                ? Colors.blue.shade800
                                : (isFilled
                                    ? Colors.green.shade800
                                    : Colors.black),
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            border: Border.all(color: Colors.grey),
          ),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 10)),
      ],
    );
  }
}

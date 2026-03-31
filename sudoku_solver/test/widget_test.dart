// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';

import 'package:sudoku_solver/screens/home_screen.dart';

void main() {
  testWidgets('App shows Sudoku home controls', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(home: HomeScreen(initializeModel: false)),
    );

    expect(find.text('Sudoku Solver'), findsOneWidget);
    expect(find.text('Camera'), findsOneWidget);
    expect(find.text('Overlay'), findsOneWidget);
    expect(find.text('Scan with Camera'), findsOneWidget);
    expect(find.text('Import from Gallery'), findsOneWidget);
  });
}

package com.example.sudoku_solver

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Path
import android.os.Build
import android.view.accessibility.AccessibilityEvent

class SudokuAccessibilityService : AccessibilityService() {

	companion object {
		@Volatile
		private var activeInstance: SudokuAccessibilityService? = null

		fun performTapDirect(x: Int, y: Int): Boolean {
			return activeInstance?.performTapGesture(x, y) == true
		}
	}

    private val tapReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == "com.example.sudoku_solver.PERFORM_TAP") {
                val x = intent.getIntExtra("x", 0)
                val y = intent.getIntExtra("y", 0)
                performTapGesture(x, y)
            }
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
		activeInstance = this

        // Register receiver for tap requests
        val filter = IntentFilter("com.example.sudoku_solver.PERFORM_TAP")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(tapReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(tapReceiver, filter)
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // We don't need to track accessibility events for this use case
    }

    override fun onInterrupt() {
        // Handle interruption
    }

    private fun performTapGesture(x: Int, y: Int): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            return false
        }

        val path = Path()
        path.moveTo(x.toFloat(), y.toFloat())

        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 100))
            .build()

        return dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                super.onCompleted(gestureDescription)
                android.util.Log.d("SudokuAccessibility", "Tap completed at ($x, $y)")
            }

            override fun onCancelled(gestureDescription: GestureDescription?) {
                super.onCancelled(gestureDescription)
                android.util.Log.e("SudokuAccessibility", "Tap cancelled at ($x, $y)")
            }
        }, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (activeInstance === this) {
			activeInstance = null
		}
        try {
            unregisterReceiver(tapReceiver)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}

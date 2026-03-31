package com.example.sudoku_solver

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

class MainActivity : FlutterActivity() {
	private val channelName = "sudoku_solver/native"
	private val mediaProjectionRequestCode = 6112
	private var pendingScreenCaptureResult: MethodChannel.Result? = null

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)

		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
			.setMethodCallHandler { call: MethodCall, result: MethodChannel.Result ->
				when (call.method) {
					"isScreenCaptureReady" -> {
						result.success(ScreenCaptureService.mediaProjectionData != null)
					}

					"isAccessibilityServiceEnabled" -> {
						result.success(isAccessibilityServiceEnabled())
					}

					"requestOverlayPermission" -> requestOverlayPermission(result)
					"requestScreenCapturePermission" -> requestScreenCapturePermission(result)
					"startOverlayService" -> {
						startOverlayService()
						result.success(true)
					}

					"stopOverlayService" -> {
						stopOverlayService()
						result.success(true)
					}

					"captureScreen" -> {
						captureScreen(result)
					}

					"fillCell" -> {
						val x = call.argument<Int>("x") ?: 0
						val y = call.argument<Int>("y") ?: 0
						val value = call.argument<Int>("value") ?: 0
						fillCell(x, y, value, result)
					}

					"openAccessibilitySettings" -> {
						openAccessibilitySettings(result)
					}

					"performTap" -> {
						val x = call.argument<Int>("x") ?: 0
						val y = call.argument<Int>("y") ?: 0
						performTap(x, y, result)
					}

					else -> result.notImplemented()
				}
			}
	}

	private fun requestOverlayPermission(result: MethodChannel.Result) {
		if (Settings.canDrawOverlays(this)) {
			result.success(true)
			return
		}

		val intent = Intent(
			Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
			Uri.parse("package:$packageName")
		)
		startActivity(intent)
		result.success(false)
	}

	private fun requestScreenCapturePermission(result: MethodChannel.Result) {
		if (pendingScreenCaptureResult != null) {
			result.error("in_progress", "A permission request is already in progress.", null)
			return
		}

		val projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
		pendingScreenCaptureResult = result
		startActivityForResult(
			projectionManager.createScreenCaptureIntent(),
			mediaProjectionRequestCode
		)
	}

	private fun startOverlayService() {
		val intent = Intent(this, ScreenCaptureService::class.java)
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			ContextCompat.startForegroundService(this, intent)
		} else {
			startService(intent)
		}
	}

	private fun stopOverlayService() {
		val intent = Intent(this, ScreenCaptureService::class.java)
		stopService(intent)
	}

	override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		super.onActivityResult(requestCode, resultCode, data)
		if (requestCode == mediaProjectionRequestCode) {
			if (resultCode == Activity.RESULT_OK && data != null) {
				// Store the MediaProjection data for the service to use
				ScreenCaptureService.mediaProjectionResultCode = resultCode
				ScreenCaptureService.mediaProjectionData = data
				pendingScreenCaptureResult?.success(true)
				android.util.Log.d("MainActivity", "Screen capture permission granted")
			} else {
				pendingScreenCaptureResult?.success(false)
				android.util.Log.w("MainActivity", "Screen capture permission denied")
			}
			pendingScreenCaptureResult = null
		}
	}

	private fun captureScreen(result: MethodChannel.Result) {
		if (ScreenCaptureService.mediaProjectionData == null) {
			android.util.Log.w("MainActivity", "captureScreen requested before MediaProjection permission")
			result.success("")
			return
		}

		val screenshotFile = File(cacheDir, "screenshot.png")
		val legacyScreenshotFile = File(cacheDir, "screenshot.jpg")
		if (screenshotFile.exists()) {
			screenshotFile.delete()
		}
		if (legacyScreenshotFile.exists()) {
			legacyScreenshotFile.delete()
		}

		// Broadcast to the service to capture screen
		val intent = Intent("com.example.sudoku_solver.CAPTURE_SCREEN")
		intent.setPackage(packageName)
		sendBroadcast(intent)

		// Return the expected path after giving the service enough time to finish.
		android.os.Handler(mainLooper).postDelayed({
			if (screenshotFile.exists() && screenshotFile.length() > 0) {
				result.success(screenshotFile.absolutePath)
			} else if (legacyScreenshotFile.exists() && legacyScreenshotFile.length() > 0) {
				result.success(legacyScreenshotFile.absolutePath)
			} else {
				android.util.Log.e("MainActivity", "captureScreen did not produce screenshot file")
				result.success("")
			}
		}, 1300)
	}

	private fun fillCell(x: Int, y: Int, value: Int, result: MethodChannel.Result) {
		// Send broadcast to overlay service to perform the tap
		val intent = Intent("com.example.sudoku_solver.FILL_CELL")
		intent.setPackage(packageName)
		intent.putExtra("x", x)
		intent.putExtra("y", y)
		intent.putExtra("value", value)
		sendBroadcast(intent)
		result.success(null)
	}

	private fun openAccessibilitySettings(result: MethodChannel.Result) {
		try {
			val intent = Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS)
			intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			startActivity(intent)
			result.success(true)
			android.util.Log.d("MainActivity", "Opened Accessibility Settings")
		} catch (e: Exception) {
			android.util.Log.e("MainActivity", "Error opening Accessibility Settings", e)
			result.success(false)
		}
	}

	private fun isAccessibilityServiceEnabled(): Boolean {
		return try {
			val enabledServices = Settings.Secure.getString(
				contentResolver,
				Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
			) ?: return false

			val packageLower = packageName.lowercase(Locale.US)
			val classNameLower = SudokuAccessibilityService::class.java.name.lowercase(Locale.US)
			val shortNameLower = ".${SudokuAccessibilityService::class.java.simpleName.lowercase(Locale.US)}"

			enabledServices.split(':').any { service ->
				val normalized = service.trim().lowercase(Locale.US)
				normalized == "$packageLower/$classNameLower" ||
					normalized == "$packageLower/$shortNameLower" ||
					normalized.endsWith("/$classNameLower") ||
					normalized.endsWith("/$shortNameLower")
			}
		} catch (_: Exception) {
			false
		}
	}

	private fun performTap(x: Int, y: Int, result: MethodChannel.Result) {
		if (!isAccessibilityServiceEnabled()) {
			android.util.Log.e("MainActivity", "Accessibility service is not enabled; tap blocked")
			result.success(false)
			return
		}

		val didDispatchDirect = SudokuAccessibilityService.performTapDirect(x, y)
		if (didDispatchDirect) {
			android.util.Log.d("MainActivity", "Direct tap dispatch accepted: ($x, $y)")
			result.success(true)
			return
		}

		// Send broadcast to accessibility service to perform tap
		val intent = Intent("com.example.sudoku_solver.PERFORM_TAP")
		intent.setPackage(packageName)
		intent.putExtra("x", x)
		intent.putExtra("y", y)
		sendBroadcast(intent)
		android.util.Log.d("MainActivity", "Sent tap broadcast: ($x, $y)")
		result.success(true)
	}
}

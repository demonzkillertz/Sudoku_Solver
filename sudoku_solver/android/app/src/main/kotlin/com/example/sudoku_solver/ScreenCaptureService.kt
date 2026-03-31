package com.example.sudoku_solver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import java.io.FileOutputStream
import java.nio.ByteBuffer

class ScreenCaptureService : Service() {
    private val channelId = "sudoku_solver_capture"
    private val notificationId = 3124
    private val TAG = "ScreenCaptureService"

    private var mediaProjection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private val handler = Handler(Looper.getMainLooper())

    companion object {
        var mediaProjectionData: Intent? = null
        var mediaProjectionResultCode: Int = 0
    }

    private val captureReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                "com.example.sudoku_solver.CAPTURE_SCREEN" -> {
                    captureScreen()
                }
                "com.example.sudoku_solver.FILL_CELL" -> {
                    val x = intent.getIntExtra("x", 0)
                    val y = intent.getIntExtra("y", 0)
                    val value = intent.getIntExtra("value", 0)
                    performTap(x, y, value)
                }
            }
        }
    }

    private fun cleanupCaptureResources() {
        virtualDisplay?.release()
        virtualDisplay = null
        imageReader?.close()
        imageReader = null
    }

    private fun tryAcquireAndSaveImage(attempt: Int = 0) {
        try {
            val image = imageReader?.acquireLatestImage()
            if (image == null) {
                if (attempt < 10) {
                    handler.postDelayed({ tryAcquireAndSaveImage(attempt + 1) }, 130)
                } else {
                    Log.e(TAG, "Failed to acquire image after retries")
                    cleanupCaptureResources()
                }
                return
            }

            val bitmap = imageToBitmap(image)
            image.close()

            val screenshotPath = "${cacheDir}/screenshot.png"
            FileOutputStream(screenshotPath).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            bitmap.recycle()

            Log.d(TAG, "Screenshot saved to: $screenshotPath")
            cleanupCaptureResources()
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing screenshot", e)
            cleanupCaptureResources()
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        // Register broadcast receiver
        val filter = IntentFilter()
        filter.addAction("com.example.sudoku_solver.CAPTURE_SCREEN")
        filter.addAction("com.example.sudoku_solver.FILL_CELL")

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(captureReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(captureReceiver, filter)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = Notification.Builder(this, channelId)
            .setContentTitle("Sudoku Solver Active")
            .setContentText("Screen capture and automation ready")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .build()

        startForeground(notificationId, notification)

        // Initialize MediaProjection if data is available
        if (mediaProjectionData != null) {
            initMediaProjection()
        }

        return START_STICKY
    }

    private fun initMediaProjection() {
        try {
            val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = projectionManager.getMediaProjection(mediaProjectionResultCode, mediaProjectionData!!)
            Log.d(TAG, "MediaProjection initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaProjection", e)
        }
    }

    private fun captureScreen() {
        try {
            if (mediaProjection == null) {
                Log.w(TAG, "MediaProjection not initialized")
                initMediaProjection()
                if (mediaProjection == null) {
                    Log.e(TAG, "Failed to initialize MediaProjection")
                    return
                }
            }

            val windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val metrics = DisplayMetrics()

            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(metrics)

            val width = metrics.widthPixels
            val height = metrics.heightPixels
            val density = metrics.densityDpi

            Log.d(TAG, "Screen dimensions: ${width}x${height} @ ${density}dpi")

            // Create ImageReader
            imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)

            // Create VirtualDisplay
            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "SudokuSolverCapture",
                width, height, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null, null
            )

            // Delay and retry because first image frame can arrive late on some devices.
            handler.postDelayed({
                tryAcquireAndSaveImage()
            }, 220)

        } catch (e: Exception) {
            Log.e(TAG, "Error in captureScreen", e)
        }
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val buffer: ByteBuffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * image.width

        val bitmap = Bitmap.createBitmap(
            image.width + rowPadding / pixelStride,
            image.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)

        return Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
    }

    private fun performTap(x: Int, y: Int, value: Int) {
        // Send broadcast to accessibility service to tap the cell
        val tapIntent = Intent("com.example.sudoku_solver.PERFORM_TAP")
        tapIntent.setPackage(packageName)
        tapIntent.putExtra("x", x)
        tapIntent.putExtra("y", y)
        sendBroadcast(tapIntent)

        Log.d(TAG, "Requested tap at ($x, $y) for value $value")

        // After tapping the cell, tap the number selector (matches main.py logic)
        handler.postDelayed({
            val numberCoordinates = getNumberCoordinates(value)
            if (numberCoordinates != null) {
                val numIntent = Intent("com.example.sudoku_solver.PERFORM_TAP")
                numIntent.setPackage(packageName)
                numIntent.putExtra("x", numberCoordinates.first)
                numIntent.putExtra("y", numberCoordinates.second)
                sendBroadcast(numIntent)
                Log.d(TAG, "Requested number tap for value $value at (${numberCoordinates.first}, ${numberCoordinates.second})")
            }
        }, 150)
    }

    private fun getNumberCoordinates(number: Int): Pair<Int, Int>? {
        // Number selector coordinates matching main.py
        return when (number) {
            1 -> Pair(110, 1900)
            2 -> Pair(320, 1900)
            3 -> Pair(540, 1900)
            4 -> Pair(760, 1900)
            5 -> Pair(970, 1900)
            6 -> Pair(110, 2050)
            7 -> Pair(320, 2050)
            8 -> Pair(540, 2050)
            9 -> Pair(760, 2050)
            else -> null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            unregisterReceiver(captureReceiver)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        Log.d(TAG, "Service destroyed")
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }

        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            channelId,
            "Sudoku Solver Automation",
            NotificationManager.IMPORTANCE_LOW
        )
        channel.description = "Enables screen capture and automated solving"
        manager.createNotificationChannel(channel)
    }
}

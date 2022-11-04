package com.opencv.controller;
import android.content.Context;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;

class MyGLSurfaceView extends GLSurfaceView {

    public MyGLRenderer gMyRenderer;

    private static MainActivity mainActivity;

    public MyGLSurfaceView(Context context){
        super(context);

        setEGLContextClientVersion(2);
        setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        getHolder().setFormat(PixelFormat.RGBA_8888);
        gMyRenderer = new MyGLRenderer();
        mainActivity = new MainActivity();

        // Set the Renderer for drawing on the GLSurfaceView
        setRenderer(gMyRenderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        setZOrderMediaOverlay(true);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {

        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                mainActivity.setTouchEvent(MotionEvent.ACTION_DOWN);
                    break;
            case MotionEvent.ACTION_UP:
                mainActivity.setTouchEvent(MotionEvent.ACTION_UP);
                break;
        }
        return true;
    }

}
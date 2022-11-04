package com.opencv.controller;


import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;

public class MyGLRenderer implements GLSurfaceView.Renderer {
    private static final String TAG = "opencv";
    private Cube mCube;
    private Line mLine_LX;
    private Line mLine_LY;
    private Line mLine_LZ;

    //https://www.charlezz.com/?p=934
    float[] gRotationMatrix_L = {
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f };
    float[] gDirectionMatrix = {
            0f, 0f, 0f
    };
    float[]  gRvec = {
            0f, 0f, 0f
    };
    float[]  gTvec = {
            0f, 0f, 0f
    };

    // vPMatrix is an abbreviation for "Model View Projection Matrix"
    private final float[] vPMatrix = new float[16];
    //projectionMatrix : 카메라가 모델을 볼수 있는 공간
    private final float[] projectionMatrix = new float[16];
    //모델을 바라보는 대상
    private final float[] viewMatrix = new float[16];
    private final float[] ModelMatrix = new float[16];

    private final float[] vPMatrix_L = new float[16];

    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        // Set the background frame color
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, .0f);

        mCube = new Cube();

        mLine_LX = new Line(
                new float[]{0, 0, 0, 5, 0, 0},
                new float[]{1.f, 0.f, 0.f, 1.f}
        );
        mLine_LY = new Line(
                new float[]{0, 0, 0, 0, 5, 0},
                new float[]{0.f, 1.f, 0.f, 1.f}
        );
        mLine_LZ = new Line(
                new float[]{0, 0, 0, 0, 0, 5},
                new float[]{0.f, 0.f, 1.f, 1.f}
        );
    }

    public void onDrawFrame(GL10 unused) {
        // Redraw background color
        float[] mMVPMatrix = new float[16];
        float[] mRotationMatrix = new float[16];
        float[] scratch = new float[16];
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        //https://www.charlezz.com/?p=960
        Matrix.setLookAtM(viewMatrix, 0,
                0, 0, -10.0f,
                0, 0, 0,
                0.0f, 1.0f, 0.0f);

        Matrix.multiplyMM(vPMatrix_L, 0, viewMatrix, 0, gRotationMatrix_L,0);
        Matrix.multiplyMM(vPMatrix_L, 0, projectionMatrix, 0, vPMatrix_L, 0);

   //     http://egloos.zum.com/EireneHue/v/984622
  //      http://www.opengl-tutorial.org/kr/beginners-tutorials/tutorial-3-matrices/#%ED%9A%8C%EC%A0%84-%EB%A7%A4%ED%8A%B8%EB%A6%AD%EC%8A%A4

        /*
        Log.i(TAG, "gRotationMatrix_L00 : " + gRotationMatrix_L[0] + ", " + gRotationMatrix_L[1] + ", " + gRotationMatrix_L[2] + ", " + gRotationMatrix_L[3]
                + "\ngRotationMatrix_L10 : " + gRotationMatrix_L[4] + ", " + gRotationMatrix_L[5] + ", " + gRotationMatrix_L[6] + ", " + gRotationMatrix_L[7]
                + "\ngRotationMatrix_L20 : " + gRotationMatrix_L[8] + ", " + gRotationMatrix_L[9] + ", " + gRotationMatrix_L[10] + ", " + gRotationMatrix_L[11]
                + "\ngRotationMatrix_L30 : " + gRotationMatrix_L[12] + ", " + gRotationMatrix_L[13] + ", " + gRotationMatrix_L[14] + ", " + gRotationMatrix_L[15]
        );
        Log.i(TAG, "viewMatrix : " + viewMatrix[0] + ", " + viewMatrix[1] + ", " + viewMatrix[2] + ", " + viewMatrix[3]
                + "\nviewMatrix1 : " + viewMatrix[4] + ", " + viewMatrix[5] + ", " + viewMatrix[6] + ", " + viewMatrix[7]
                + "\nviewMatrix2 : " + viewMatrix[8] + ", " + viewMatrix[9] + ", " + viewMatrix[10] + ", " + viewMatrix[11]
                + "\nviewMatrix3 : " + viewMatrix[12] + ", " + viewMatrix[13] + ", " + viewMatrix[14] + ", " + viewMatrix[15]
        );
        Log.i(TAG, "projectionMatrix : " + projectionMatrix[0] + ", " + projectionMatrix[1] + ", " + projectionMatrix[2] + ", " + projectionMatrix[3]
                + "\nprojectionMatrix1 : " + projectionMatrix[4] + ", " + projectionMatrix[5] + ", " + projectionMatrix[6] + ", " + projectionMatrix[7]
                + "\nprojectionMatrix2 : " + projectionMatrix[8] + ", " + projectionMatrix[9] + ", " + projectionMatrix[10] + ", " + projectionMatrix[11]
                + "\nprojectionMatrix3 : " + projectionMatrix[12] + ", " + projectionMatrix[13] + ", " + projectionMatrix[14] + ", " + projectionMatrix[15]
        );
        Log.i(TAG, "vPMatrix_L0 : " + vPMatrix_L[0] + ", " + vPMatrix_L[1] + ", " + vPMatrix_L[2] + ", " + vPMatrix_L[3]
                + "\nvPMatrix_L1 : " + vPMatrix_L[4] + ", " + vPMatrix_L[5] + ", " + vPMatrix_L[6] + ", " + vPMatrix_L[7]
                + "\nvPMatrix_L2 : " + vPMatrix_L[8] + ", " + vPMatrix_L[9] + ", " + vPMatrix_L[10] + ", " + vPMatrix_L[11]
                + "\nvPMatrix_L3 : " + vPMatrix_L[12] + ", " + vPMatrix_L[13] + ", " + vPMatrix_L[14] + ", " + vPMatrix_L[15]
        );
*/


        mLine_LX.draw(vPMatrix_L);
        mLine_LY.draw(vPMatrix_L);
        mLine_LZ.draw(vPMatrix_L);
        //projection matrix와 camera view matrix를 곱하여 mMVPMatrix 변수에 저장
        Matrix.multiplyMM(mMVPMatrix, 0, projectionMatrix, 0, viewMatrix, 0);
        //mAngle 각도값을 이용하여 (x,y,z)=(1,1,1) 축 주위를 회전하는 회전 matrix를 정의합니다.
        //Matrix.setRotateM(mRotationMatrix, 0, mAngle, 1, 1, 1);
        //projection matrix와 camera view matrix를 곱하여 얻은 matrix인 mMVPMatrix와
        //회전 matrix mRotationMatrix를 결합합니다.
        Matrix.multiplyMM(scratch, 0, mMVPMatrix, 0, gRotationMatrix_L, 0);
        mCube.draw(scratch);

    }

    public void onSurfaceChanged(GL10 unused, int width, int height) {
        GLES20.glViewport(0, 0, width, height);

        float ratio = (float) width / height;
        Log.i(TAG, "w/h ratio, " + ratio);

        //perspective
        Matrix.perspectiveM(projectionMatrix,0 , 60, ratio, 1, 30);
   //     Matrix.frustumM(projectionMatrix, 0, -ratio, ratio, -1, 1, 1, 7);
    }

    public static int loadShader(int type, String shaderCode){

        // create a vertex shader type (GLES20.GL_VERTEX_SHADER)
        // or a fragment shader type (GLES20.GL_FRAGMENT_SHADER)
        int shader = GLES20.glCreateShader(type);

        // add the source code to the shader and compile it
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);

        return shader;
    }

    //System.arraycopy는 byte[] 형태의 데어티를 자르거나 붙일때 씀
    //System.arraycopy(src, srcPos, dest, destPos, lenght);
    //src : 복사하고자 하는 src, 원본
    //srcPos : src의 어디서부터 읽어올지 정한다., 처음부터는 0
    //dest : 복사할 소스
    //destPost : 복사할 소스 위치
    //length : 길이

    public void setRMatCamera(float[] R){
        System.arraycopy(R, 0, gRotationMatrix_L, 0, 16);
    }

    public void setRMatCameraOne(float[] R){
        System.arraycopy(R, 0, gRotationMatrix_L, 0, 16);
    }

    public void setDirectionMat(float[] direction) {
        System.arraycopy(direction, 0, gDirectionMatrix, 0, 3);
    }

    public void setRVecTVec(float[] Rvec, float[] Tvec) {
        System.arraycopy(Rvec, 0, gRvec, 0, 3);
        System.arraycopy(Tvec, 0, gTvec, 0, 3);
    }

    public void setRMatGyro(float[] r) {
    }

    public volatile float mAngle;
    public float getAngle() {
        return mAngle;
    }
    public void setAngle(float angle) {
        mAngle = angle;
    }
}
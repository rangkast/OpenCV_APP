package com.opencv.controller;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/*
https://kpage.tistory.com/231
https://webnautes.tistory.com/1054?category=704164
 */

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2, SensorEventListener {

    private static final String TAG = "opencv";

    private Mat matInput;
    private Mat matResult;

    public static CameraBridgeViewBase mOpenCvCameraView;

    public native int ConvertRGBtoGray(long matAddrInput, long matAddrResult, float[] matAddrRotation, int[] cInfo);
    public native void GetRvecTvec(float[] mat_addr_rvec, float[] mat_addr_info);
    public native void NativeInfo(float[] mat_addr_tvec);
    public native void InitMatrix();

    static int bDetectObj = 0;

    // bluetooth
    private BluetoothService btService = null;
    private final Handler mHandler = new Handler(){
        @Override public void handleMessage(Message msg){
            super.handleMessage(msg);
        }
    };
    ArrayList<String> permissions = new ArrayList<>();
    ArrayList hasPermissions = new ArrayList();
    ArrayList permissonsRationale = new ArrayList();
    // Intent request code private static
    final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;
    String strBTState = "";

    private static int mTouchStatus = 1;
    // opengl
    private static MyGLSurfaceView gLView;
    public float[] gRMat = new float[32];
    //transfer datas for native
    public int[] cInfo = new int[4];
    public float[] gRvec = new float[3];
    public float[] gTvec = new float[3];
    public float[] native_info = new float[16];
    // Sensor
    private SensorManager mSensorManager;
    private Sensor mGyro;
    private Sensor mAccel;
    //Roll and Pitch
    private double pitch;
    private double roll;
    private double yaw;
    //timestamp and dt
    private double timestamp;
    private double dt;
    // for radian -> dgree
    private double RAD2DGR = 180 / Math.PI;
    private static final float NS2S = 1.0f/1000000000.0f;
    // Sensor Calculate class
    MySensor mMySensor;
    TextView java_textview;
    TextView jv_row1;
    TextView jv_row2;
    TextView jv_row3;
    TextView jv_row4;
    TextView native_textview;
    TextView nv_row1;
    TextView nv_row2;
    TextView nv_row3;
    TextView nv_row4;

    private static LayoutInflater controlInflater;
    private static View viewControl;
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.d(TAG, "onCreate");
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        // opengl
        gLView = new MyGLSurfaceView(this);
        addContentView(gLView, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        Log.d(TAG, "init OpenGL");
        //add Permission to ArrayList
        permissions.add(Manifest.permission.CAMERA);
        checkPermissions();

//      mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) new JavaCameraView(this, -1);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        addContentView(mOpenCvCameraView, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        controlInflater = LayoutInflater.from(this);
        viewControl = controlInflater.inflate(R.layout.control, null);

        addContentView(viewControl, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        java_textview = (TextView) viewControl.findViewById(R.id.text_java);
        java_textview.setText("java_text view");
        jv_row1 = (TextView) viewControl.findViewById(R.id.jv_row1);
        jv_row2 = (TextView) viewControl.findViewById(R.id.jv_row2);
        jv_row3 = (TextView) viewControl.findViewById(R.id.jv_row3);
        jv_row4 = (TextView) viewControl.findViewById(R.id.jv_row4);
        native_textview = (TextView) viewControl.findViewById(R.id.text_native);
        native_textview.setText("native_text view");
        nv_row1 = (TextView) viewControl.findViewById(R.id.nv_row1);
        nv_row2 = (TextView) viewControl.findViewById(R.id.nv_row2);
        nv_row3 = (TextView) viewControl.findViewById(R.id.nv_row3);
        nv_row4 = (TextView) viewControl.findViewById(R.id.nv_row4);
        // NDK Rotation Matrix initialize
        InitMatrix();
        // Example of a call to a native method
        //textView.setText(stringFromJNI());



        // Bluetooth
        if (true) { // for GL TEST
            if (btService == null) {
                Log.i(TAG, "try new BluetoothService");
                btService = new BluetoothService(this, mHandler);
            }

			if(btService.mAdapter == null)
            {
                Toast.makeText(this, "Bluetooth is not available", Toast.LENGTH_LONG).show();
                finish();
                return;
            }
        }

        // Sensor
        /*
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mGyro = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        //linear acceleration = acceleration - acceleration due to gravity
        mAccel = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mMySensor = new MySensor();
         */
    }

    @Override
    public void onStart() {
        super.onStart();
        boolean havePermission = true;

        if (havePermission) {
            onCameraPermissionGranted();
        }

		if (!btService.mAdapter.isEnabled()) {
            Intent enableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(enableIntent, REQUEST_ENABLE_BT);
        // Otherwise, setup the chat session
        } else {
            Log.i(TAG, "try new BluetoothService");
            btService = new BluetoothService(this, mHandler);
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        if (gLView != null)
            gLView.onPause();
 //       if (mSensorManager != null)
 //           mSensorManager.unregisterListener(this);
//        if (btService == null)
//            btService.stop();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        checkPermissions();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        if (gLView != null) {
            gLView.onResume();
        }

        /*
        if (mSensorManager != null){
            mSensorManager.registerListener(this, mGyro, SensorManager.SENSOR_DELAY_UI);
            mSensorManager.registerListener(this, mAccel, SensorManager.SENSOR_DELAY_UI);
        } else {
            Log.d(TAG, "onResum :: mSensorManager in NULL");
        }
        */

        if (btService != null) {
            // Only if the state is STATE_NONE, do we know that we haven't started already
            if (btService.getState() == BluetoothService.STATE_NONE) {
                // Start the Bluetooth chat services
                btService.start();
            }
        } else {
            btService = new BluetoothService(this, mHandler);
            if(btService.mAdapter == null)
            {
                Toast.makeText(this, "Bluetooth is not available", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // Stop the Bluetooth chat services
        if (btService == null)
            btService.stop();
			
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
			
        if (gLView != null)
            gLView.onPause();
			
  //      if (mSensorManager != null)
  //          mSensorManager.unregisterListener(this);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    if (mOpenCvCameraView != null)
                        mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
//    public native String stringFromJNI();

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //camera input frame에서 Mat 정보를 얻을 수 있다.
        matInput = inputFrame.rgba();
        if ( matResult != null )
            matResult.release();

        matResult = new Mat(matInput.rows(), matInput.cols(), matInput.type());
        /*
         Log.d(TAG, "matResult " + matInput.rows() + " " + matInput.cols() + " " + matInput.type());
         720(row) * 1600(col) : 해상도
         24 (type) CV_8UC4 (8bit unsigned Channel 4)
         rgba(Red, Green, Blue, Alpha(transparent))
          */
         cInfo[0] = matInput.rows();
         cInfo[1] = matInput.cols();
         cInfo[2] = 90;
         cInfo[3] = mTouchStatus;

         bDetectObj = ConvertRGBtoGray(matInput.getNativeObjAddr(), matResult.getNativeObjAddr(), gRMat, cInfo);


    /*
           Log.d(TAG, "Main cInfo " + " " + cInfo[0] + " " + cInfo[1] + " " + cInfo[2] + " " +cInfo[3]);
           GetRvecTvec(gRvec, gTvec);
           Log.i(TAG, "GetRvecTvec, "
                           + gRvec[0] + ", " + gRvec[1] + ", " + gRvec[2] + ", gTvec, "
                           + gTvec[0] + ", " + gTvec[1] + ", " + gTvec[2] + ", bRes, " + bDetectObj);

           Log.i(TAG, "onCameraFrame: \n"
                   + gRMat[0] + ", " + gRMat[1] + ", " + gRMat[2] + ", " + gRMat[3]
                   + "\n" + gRMat[4] + ", " + gRMat[5] + ", " + gRMat[6] + ", " + gRMat[7]
                   + "\n" + gRMat[8] + ", " + gRMat[9] + ", " + gRMat[10] + ", " + gRMat[11]
                   + "\n" + gRMat[12] + ", " + gRMat[13] + ", " + gRMat[14] + ", " + gRMat[15]
           );

            Log.i(TAG, "bDetectObj " + bDetectObj);
    */
        NativeInfo(native_info);
        /*
        Log.d(TAG, "rv:\n"
                + native_info[0] + ", " + native_info[1] + ", " + native_info[2] + "\n"
                + native_info[3] + ", " + native_info[4] + ", " + native_info[5] + "\n"
                + native_info[6] + ", " + native_info[7] + ", " + native_info[8] + "\n"
        );

         */
        new Thread(new Runnable() {
            @Override public void run() {
                // 현재 UI 스레드가 아니기 때문에 메시지 큐에 Runnable을 등록 함
                runOnUiThread(new Runnable() {
                    public void run() {
                        StringBuilder str = new StringBuilder();
                        str.append(cInfo[0] + " x " + cInfo[1] + " " + " touchStatus: " + cInfo[3] + "\n");
                        java_textview.setText(str.toString());
                        jv_row1.setText(
                                String.format("%.3f",gRMat[0]) + "\n"
                                + String.format("%.3f",gRMat[4]) + "\n"
                                + String.format("%.3f",gRMat[8]) + "\n"
                                + String.format("%.3f",gRMat[12]) + "\n");
                        jv_row2.setText(
                                String.format("%.3f",gRMat[1])+ "\n"
                                + String.format("%.3f",gRMat[5]) + "\n"
                                + String.format("%.3f",gRMat[9]) + "\n"
                                + String.format("%.3f",gRMat[13]) + "\n");
                        jv_row3.setText(
                                String.format("%.3f",gRMat[2]) + "\n"
                                + String.format("%.3f",gRMat[6]) + "\n"
                                + String.format("%.3f",gRMat[10]) + "\n"
                                + String.format("%.3f",gRMat[14]) + "\n");
                        jv_row4.setText(
                                String.format("%.3f",gRMat[3]) + "\n"
                                + String.format("%.3f",gRMat[7]) + "\n"
                                + String.format("%.3f",gRMat[11]) + "\n"
                                + String.format("%.3f",gRMat[15]) + "\n");


                        str.setLength(0);

                        str.append("start: "  + native_info[13] + "\ncontours: " +  native_info[12]
                                + "\nsolutions: " + native_info[14]);
                        native_textview.setText(str.toString());

                        nv_row2.setText("R_Vector\n"
                                        + String.format("%.3f",native_info[0]) + "\n"
                                        + String.format("%.3f",native_info[3]) + "\n"
                                        + String.format("%.3f",native_info[6]) + "\n"
                                        + "T_Vector\n"
                                        + String.format("%.3f",native_info[9]) + "\n");
                        nv_row3.setText("\n"
                                        + String.format("%.3f",native_info[1])+ "\n"
                                        + String.format("%.3f",native_info[4]) + "\n"
                                        + String.format("%.3f",native_info[7]) + "\n"
                                        + "\n"
                                        + String.format("%.3f",native_info[10]) + "\n");

                        nv_row4.setText("\n"
                                        + String.format("%.3f",native_info[2]) + "\n"
                                        + String.format("%.3f",native_info[5]) + "\n"
                                        + String.format("%.3f",native_info[8]) + "\n"
                                        + "\n"
                                        + String.format("%.3f",native_info[11]) + "\n");
                    }
                });
            }
        }).start();



/*
        if (bDetectObj == 2) {
            gLView.gMyRenderer.setRMatCamera(gRMat);      // Greendot rotation matrix
            gLView.gMyRenderer.setRVecTVec(gRvec, gTvec);       //Greendot rvec, tvec
        }
        else
 */
        if (bDetectObj == 1){
            gLView.gMyRenderer.setRMatCameraOne(gRMat);
        }

        return matResult;
    }


    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    //여기부터는 런타임 퍼미션 처리을 위한 메소드들
    @TargetApi(Build.VERSION_CODES.M)
    private void checkPermissions() {
        int permissonCnt = 0;
        Boolean needPermissionRequest = false;
        Boolean needRationaleRequest = false;

        permissonCnt = permissions.size();

        //arraylist to string array
        String[] arrayPermissions = permissions.toArray(new String[permissonCnt]);

        for (int i = 0; i < permissonCnt; i++) {
            Object obj = permissions.get(i);
            if (obj instanceof String) {
                String str = (String) obj;
                permissonsRationale.add(new Boolean(ActivityCompat.shouldShowRequestPermissionRationale(this, str)));
                hasPermissions.add(new Integer(ContextCompat.checkSelfPermission(this, str)));
                //      Log.d(TAG, " " + str + ": " + hasPermissions.get(i) + ", " + permissonsRationale.get(i));

                if ((Integer) hasPermissions.get(i) == PackageManager.PERMISSION_DENIED) {
                    needPermissionRequest = true;
                }

                if ((Boolean) permissonsRationale.get(i) == false) {
                    needRationaleRequest = true;
                }
            }
            //    Log.d(TAG,"requestPermissionString: " + arrayPermissions[i]);
        }

        if (needPermissionRequest) {
            ActivityCompat.requestPermissions(this, arrayPermissions, 0);
            //     Log.d(TAG,"requesting...");
        }

        if ((Integer) hasPermissions.get(0) == PackageManager.PERMISSION_GRANTED) {
        }
    }

    protected void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            onCameraPermissionGranted();
        } else{
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    /**
     * A native method that is implemented by the 'controller' native library,
     * which is packaged with this application.
     */
//dell    public native String stringFromJNI();

    // Bluetooth
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case REQUEST_CONNECT_DEVICE:
                // When DeviceListActivity returns with a device to connect
                if (resultCode == Activity.RESULT_OK) {
                    if (btService != null)
                    	btService.getDeviceInfo(data);
                }
                break;
            case REQUEST_ENABLE_BT:
                // When the request to enable Bluetooth returns
                if (resultCode == Activity.RESULT_OK) {
                    if (btService != null)
                        btService.scanDevices();
                } else {
                    Log.d(TAG, "Bluetooth is not enabled");
                }
                break;
        }
    }

    // Sensor
    private final static float SENSOR_RESOLUTION_RADIAN_PER_DEGREE = 0.1f;//0.01745f; //TODO CHECK
    float mHorizontalRadian = 0.f;
    float mVerticalRadian = 0.f;
    float mDirection[] = {0f, 0f, 0f};
    @Override
    public void onSensorChanged(SensorEvent event) {
        Sensor sensor = event.sensor;
        double gyroX = .0;
        double gyroY = .0;
        double gyroZ = .0;

        dt = (event.timestamp - timestamp) * NS2S;
        timestamp = event.timestamp;
        mMySensor.setDt(dt);

        if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroX = event.values[0];
            gyroY = event.values[1];
            gyroZ = event.values[2];
            mMySensor.setGyro(event);
//            Log.i("naomi_sensor", "xyz, " + event.values[0] + ", " + event.values[1] + ", " + event.values[2] );

            if (dt - timestamp * NS2S != 0) {
                roll = roll + gyroX * dt;
                pitch = pitch + gyroY * dt;
                yaw = yaw + gyroZ * dt;
            }
            // toQuaternion
            double cy = Math.cos(yaw * 0.5);
            double sy = Math.sin(yaw * 0.5);
            double cp = Math.cos(pitch * 0.5);
            double sp = Math.sin(pitch * 0.5);
            double cr = Math.cos(roll * 0.5);
            double sr = Math.sin(roll * 0.5);

            double q_w = cr * cp * cy + sr * sp * sy;
            double q_x = sr * cp * cy - cr * sp * sy;
            double q_y = cr * sp * cy + sr * cp * sy;
            double q_z = cr * cp * sy - sr * sp * cy;

            float posXRadian = (float)(gyroX * SENSOR_RESOLUTION_RADIAN_PER_DEGREE);
            float posYRadian = (float)(gyroY * SENSOR_RESOLUTION_RADIAN_PER_DEGREE);

            mHorizontalRadian += posYRadian;
            mVerticalRadian += posXRadian;

            mDirection[0] = (float) (Math.cos(mVerticalRadian) * Math.sin(mHorizontalRadian));
            mDirection[1] = (float) (Math.sin(mVerticalRadian));
            mDirection[2] = (float) (Math.cos(mVerticalRadian) * Math.cos(mHorizontalRadian));
            gLView.gMyRenderer.setDirectionMat(mDirection);

            Log.i(TAG, "xyz, " + String.format("%7.3f", gyroX) + ", " + String.format("%7.3f", gyroY) + ", " + String.format("%7.3f", gyroZ)
                    + ", r/p/y, " + String.format("%7.3f", roll) + ", " + String.format("%7.3f", pitch) + ", " + String.format("%7.3f", yaw)
                    + ", dt, " + String.format("%7.3f", dt)
                    + ", q_w/x/y/z, " + String.format("%7.3f", q_w) + ", " + String.format("%7.3f", q_x)
                    + ", " + String.format("%7.3f", q_y)+ ", " + String.format("%7.3f", q_z)
                    + ", directions, " + mDirection[0] + ", " + mDirection[1] + ", " + mDirection[2]
                    );

        }
        else if(sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION){
            mMySensor.setAccel(event);
        }

        mMySensor.calFixedAngle();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }
    // get bluetooth data
    // read OK xyz,   0.133,  -0.104,   0.065, r/p/y,   0.009,  -0.007,   0.004, dt,   0.067, q_w/x/y/z,   1.000,   0.004,  -0.003,   0.002
    static float mGARORadian = 0.f;
    static float mSERORadian = 0.f;
    static float mGyroDir[] = {0f, 0f, 0f};
    public static void fromControllerRvec(String strYes) {
        String[] v = strYes.split(", ");
        Log.i(TAG, "fromContorller String : " + strYes);
        if(v.length == 15){
            float[] rvec = new float[] { Float.parseFloat(v[1]), Float.parseFloat(v[2]), Float.parseFloat(v[3]), 1.f};
            Log.i(TAG, "rvec," + rvec[0]  + "," + rvec[1] + "," + rvec[2]);

            float GyroXRadian = (float)(rvec[0] * SENSOR_RESOLUTION_RADIAN_PER_DEGREE);
            float GyroYRadian = (float)(rvec[1] * SENSOR_RESOLUTION_RADIAN_PER_DEGREE);

            mGARORadian += GyroYRadian;
            mSERORadian += GyroXRadian;

            mGyroDir[0] = (float) (Math.cos(mSERORadian) * Math.sin(mGARORadian));
            mGyroDir[1] = (float) (Math.sin(mSERORadian));
            mGyroDir[2] = (float) (Math.cos(mSERORadian) * Math.cos(mGARORadian));

            float[] R = new float[16];
            SensorManager.getRotationMatrixFromVector(R, mGyroDir);
            Log.i(TAG, "RformRvec : " + R[0] + ", " + R[1] + ", " + R[2] + ", " + R[3]
                    + "\nRformRvec1 : " + R[4] + ", " + R[5] + ", " + R[6] + ", " + R[7]
                    + "\nRformRvec2 : " + R[8] + ", " + R[9] + ", " + R[10] + ", " + R[11]
                    + "\nRformRvec3 : " + R[12] + ", " + R[13] + ", " + R[14] + ", " + R[15]
            );

//            if(!bDetectObj) {
//                gLView.gMyRenderer.setRMatGyro(R);
//            }
            if(bDetectObj == 1)
                gLView.gMyRenderer.setRMatGyro(R);
        }
    }


    public void setBTstatus(String str) {
        mOpenCvCameraView.setBTStatus(str);
    }


    public void setTouchEvent (int event) {
        mTouchStatus = event;

    }
}

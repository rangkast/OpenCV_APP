package com.naomi.greendot;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.Vibrator;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import java.util.ArrayList;

public class MainActivity extends Activity implements SensorEventListener {
    private static final String TAG = "naomi";

    private SensorManager mSensorManager;

    MyView mMyView;

    private Sensor mGyro;

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

    // Status Flag
    boolean bGreen = false;

    // Bluetooth
    private BluetoothService btService = null;
    private final Handler mHandler = new Handler() {
        @Override public void handleMessage(Message msg) {
            super.handleMessage(msg);
        }
    };
    // Intent request code private static
    final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;

    String strBTstat = "";
    //

    Vibrator vibrator;
    private int long_click = 0;

    private long time = 0;
    private float event_x = 0;
    private float event_y = 0;




    //green dot setting
    private int radius = 60;
    private int times = 100;
    private int middlex = 950;
    private int middley = 1000;

    static class COORD {
        int x;
        int y;
        public COORD (int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    static ArrayList<COORD> node_list = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        Log.i(TAG, "Sensor");
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mGyro = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        node_list.add(new COORD(middlex - 3 * times, middley - 4 * times));
        node_list.add(new COORD(middlex, middley ));
        node_list.add(new COORD(middlex - 8 * times, middley + 7 * times));
        node_list.add(new COORD(middlex - 7 * times, middley -5 * times));

        mMyView = new MyView(MainActivity.this);
        mMyView.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View v, MotionEvent event) {
        //        Log.d(TAG, "x: " + event.getX() + " y: " + event.getY() + " " + event.getPointerCount());
                if (event.getPointerCount() > 1) {
                    time = 0;
                    long_click = 0;
                }

                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
//                        Log.d(TAG, "Action down");
                        if (time == 0) {
                            event_x = event.getX();
                            event_y = event.getY();
                            time =  System.currentTimeMillis();
                            long_click = 0;
                        }
                        break;
                    case MotionEvent.ACTION_UP:
            //            Log.d(TAG, "Action up");

                        if (long_click == 0) {
                            if (bGreen) {
                                bGreen = false;
                                stopSensor(v);
                                if (btService != null)
                                    btService.stopSendData();
                            }
                            else {
                                bGreen = true;
                                startSensor(v);
                                if (btService != null)
                                    btService.startSendData();
                            }
                        }
                        time = 0;

                        break;
                    case MotionEvent.ACTION_MOVE:
               //         Log.d(TAG, "Action move");
                        break;
                }

                if (time != 0) {
                    if (Math.abs(event.getX() - event_x) < 50 &&
                            Math.abs(event.getY() - event_y) < 50 &&
                            (System.currentTimeMillis() - time) > 600) {
                        vibrator.vibrate(100);
                //        Log.d(TAG, "long click");
                        long_click= 1;
                        time = 0;
                        node_list.add(new COORD((int)event_x, (int)event_y));
                    }
                }

                return true;
            }
        });

        setContentView(mMyView);

        // Bluetooth

        if (false) {
            if(btService == null) {
                btService = new BluetoothService(this, mHandler);
            }
            if(btService.getDeviceState()){
                // can use BT devices
                btService.enableBluetooth();
            } else {
                finish();
            }
        }
    }

    public void startSensor(View view){
        mSensorManager.registerListener(this, mGyro, SensorManager.SENSOR_DELAY_UI);
    }
    public void stopSensor(View view){
        mSensorManager.unregisterListener(this);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        Sensor sensor = event.sensor;
        double gyroX = .0;
        double gyroY = .0;
        double gyroZ = .0;

        if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroX = event.values[0];
            gyroY = event.values[1];
            gyroZ = event.values[2];
//            Log.i("naomi_sensor", "xyz, " + event.values[0] + ", " + event.values[1] + ", " + event.values[2] );
            dt = (event.timestamp - timestamp) * NS2S;
            timestamp = event.timestamp;

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

//            Log.i(TAG, "xyz, " + String.format("%7.3f", gyroX) + ", " + String.format("%7.3f", gyroY) + ", " + String.format("%7.3f", gyroZ)
//                    + ", r/p/y, " + String.format("%7.3f", roll) + ", " + String.format("%7.3f", pitch) + ", " + String.format("%7.3f", yaw)
//                    + ", dt, " + String.format("%7.3f", dt)
//                    + ", q_w/x/y/z, " + String.format("%7.3f", q_w) + ", " + String.format("%7.3f", q_x)
//                    + ", " + String.format("%7.3f", q_y)+ ", " + String.format("%7.3f", q_z));
            String data = new String ("xyz, " + String.format("%7.3f", gyroX) + ", " + String.format("%7.3f", gyroY) + ", " + String.format("%7.3f", gyroZ)
                    + ", r/p/y, " + String.format("%7.3f", roll) + ", " + String.format("%7.3f", pitch) + ", " + String.format("%7.3f", yaw)
                    + ", dt, " + String.format("%7.3f", dt)
                    + ", q_w/x/y/z, " + String.format("%7.3f", q_w) + ", " + String.format("%7.3f", q_x)
                    + ", " + String.format("%7.3f", q_y)+ ", " + String.format("%7.3f", q_z) + "\n");
            if (btService != null)
                btService.SetSensorDataToBT(data);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
//        mSensorManager.registerListener(this, mGyro, SensorManager.SENSOR_DELAY_FASTEST);
//        mSensorManager.registerListener(this, mGyro, SensorManager.SENSOR_DELAY_UI);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
        bGreen = false;
//        bluetoothOff();
    }
	
	@Override
    public void onResume()
    {
        super.onResume();
		
        if (btService != null) {
            // Only if the state is STATE_NONE, do we know that we haven't started already
            if (btService.getState() == BluetoothService.STATE_NONE) {
                // Start the Bluetooth chat services
                btService.start();
            }
        } else {
            btService = new BluetoothService(this, mHandler);
            if (btService.getDeviceState()) {
                // can use BT devices
                btService.enableBluetooth();
            } else {
                Log.i(TAG, "getDeviceState Fail. finish()");
                finish();
            }
        }
    }		

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (btService == null)
            btService.stop();
    }

    class MyView extends View {
        public MyView(Context context) {
            super(context); // 부모의 인자값이 있는 생성자를 호출한다
        }

        public boolean onTouchEvent(MotionEvent event){
            invalidate();
            return super.onTouchEvent(event);
        }

        @Override
        protected void onDraw(Canvas canvas) { // 화면을 그려주는 작업
            Paint paint = new Paint(); // 화면에 그려줄 도구를 셋팅하는 객체

            setBackgroundColor(Color.DKGRAY); // 배경색을 지정


            if (bGreen){
                paint.setColor(Color.GREEN);
            }
            else {
                paint.setColor(Color.RED);
            }
            for (int i = 0; i < node_list.size() ; i++)
                canvas.drawCircle(node_list.get(i).x, node_list.get(i).y, radius, paint); // 원의중심 x,y, 반지름,paint

            paint.setTextSize(40.f);
            paint.setColor(Color.YELLOW);
            if (btService.getState() == 3)
                strBTstat = "Connected";
            else
                strBTstat = "X - Cannot use";
            canvas.drawText(strBTstat, 100, 100, paint);

            invalidate();
        }


    }

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
                    // 확인 눌렀을 때
                    // Next Step
                    if (btService != null)
                        btService.scanDevices();
                } else {
                    // 취소 눌렀을 때
                    Log.d(TAG, "Bluetooth is not enabled");
                }
                break;
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
 //      Log.d(TAG, "keyevent " + event.getAction() + " keycode" + keyCode);
        switch (keyCode) {
            case KeyEvent.KEYCODE_MENU:
                break;
            case KeyEvent.KEYCODE_HOME:
                break;
            case KeyEvent.KEYCODE_BACK:
                break;
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                if (node_list.size() > 0)
                    node_list.remove(node_list.size() - 1);
                mMyView.invalidate();
                break;
            case KeyEvent.KEYCODE_VOLUME_UP:
                break;
        }
        return super.onKeyDown(keyCode, event);
    }
}
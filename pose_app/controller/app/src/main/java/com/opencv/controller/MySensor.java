package com.opencv.controller;

import android.hardware.SensorEvent;

public class MySensor {

    private static final String TAG = "opencv";

    double AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ;

    double angleAcX, angleAcY, angleAcZ;
    double angleGyX, angleGyY, angleGyZ;
    double angleFiX, angleFiY, angleFiZ;

    double averAcX, averAcY, averAcZ;
    double averGyX, averGyY, averGyZ;

    final double RADIAN_TO_DEGREE = 180 / 3.14159;
    final double DEG_PER_SEC = 32767 / 250;     // rotation angle for 1 sec
    final double ALPHA = 1 / (1 + 0.04);

    final private int INIT_DATA_NUM = 10;

    double dt;

    int gInitCnt;

    public void MySensor(){
        gInitCnt = 0;
    }

    public void calFixedAngle() {
        if (setInitValues()) {
            angleAcX = Math.atan(AcY / Math.sqrt(Math.pow(AcX, 2) + Math.pow(AcZ, 2)));
            angleAcX *= RADIAN_TO_DEGREE;
            angleAcY = Math.atan(-AcX / Math.sqrt(Math.pow(AcY, 2) + Math.pow(AcZ, 2)));
            angleAcY *= RADIAN_TO_DEGREE;

            // 가속도 현재 값에서 초기평균값을 빼서 센서값에 대한 보정
//            angleGyX += ((GyX - averGyX) / DEG_PER_SEC) * dt;  //각속도로 변환
//            angleGyY += ((GyY - averGyY) / DEG_PER_SEC) * dt;
//            angleGyZ += ((GyZ - averGyZ) / DEG_PER_SEC) * dt;
            angleGyX += (GyX - averGyX) * dt;
            angleGyY += (GyY - averGyY) * dt;
            angleGyZ += (GyZ - averGyZ) * dt;

            // 상보필터 처리를 위한 임시각도 저장
            double angleTmpX = angleFiX + angleGyX * dt;
            double angleTmpY = angleFiY + angleGyY * dt;
            double angleTmpZ = angleFiZ + angleGyZ * dt;

            // (상보필터 값 처리) 임시 각도에 0.96가속도 센서로 얻어진 각도 0.04의 비중을 두어 현재 각도를 구함.
            angleFiX = ALPHA * angleTmpX + (1.0 - ALPHA) * angleAcX;
            angleFiY = ALPHA * angleTmpY + (1.0 - ALPHA) * angleAcY;
            angleFiZ = angleGyZ;    // Z축은 자이로 센서만을 이용하열 구함.

//            Log.i(TAG, "acc raw angle, " + angleAcX + ", " + angleAcY + ", " + angleAcZ
//                    + ", acc Filtered angle, " + angleFiX + ", " + angleFiY + ", " + angleFiZ
//                    + ", gyro raw angle, " + GyX + ", " + GyY + ", " + GyZ
//                    + ", gyro filtered angle, " + angleGyX + ", " + angleGyY + ", " + angleGyZ
//            );
        }
    }

    private boolean setInitValues(){
        if(gInitCnt < INIT_DATA_NUM){
            averAcX+=AcX;  averAcY+=AcY;  averAcZ+=AcZ;
            averGyX+=GyX;  averGyY+=GyY;  averGyZ+=GyZ;
            gInitCnt ++;

            return false;
        }
        else if (gInitCnt < INIT_DATA_NUM + 1){
            averAcX /= INIT_DATA_NUM;
            averAcY /= INIT_DATA_NUM;
            averAcZ /= INIT_DATA_NUM;
            averGyX /= INIT_DATA_NUM;
            averGyY /= INIT_DATA_NUM;
            averGyZ /= INIT_DATA_NUM;
            gInitCnt ++;
        }
        return true;
    }

    public void setAccel(SensorEvent event) {
        AcX = event.values[0];
        AcY = event.values[1];
        AcZ = event.values[2];
    }

    public void setGyro(SensorEvent event) {
        GyX = event.values[0];
        GyY = event.values[1];
        GyZ = event.values[2];
    }

    public void setDt(double time_diff) {
        dt = time_diff;
    }

}

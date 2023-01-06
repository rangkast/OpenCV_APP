/*==============================================================================
            Copyright (c) 2010-2011 Qualcomm Technologies Incorporated.
            All Rights Reserved.
            Qualcomm Technologies Confidential and Proprietary
==============================================================================*/

package project.test.fastcv_app;

import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.WindowManager;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.CameraBridgeViewBase;

import java.util.ArrayList;
import java.util.List;

/** The splash screen activity for FastCV sample application */
public class SplashScreen extends Activity 
{
    protected static final String  TAG  = "[FASTCV]SplashScreen";
    private WindowManager               mWindowManager;
    private SplashScreenView            mHomeView;
    public static Display               sDisplay;
    ArrayList<String> permissions = new ArrayList<>();
    ArrayList hasPermissions = new ArrayList();
    ArrayList permissonsRationale = new ArrayList();
    // Intent request code private static
    final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;
    static
    {
        System.loadLibrary( "fastcv_app" );
    }
    
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mWindowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        sDisplay = mWindowManager.getDefaultDisplay();
      
        // Initialize UI
        mHomeView = new SplashScreenView(this);
        setContentView( mHomeView );


        //add Permission to ArrayList
        permissions.add(Manifest.permission.CAMERA);
        checkPermissions();

        String fastcvVersion = getFastCVVersion();
        Log.d(TAG, "fastcv_version"  + fastcvVersion);
    }
     
    @Override
    protected void onPause() 
    {
       super.onPause(); 
    }
    /** Called when the option menu is created. */
    @Override
    public boolean onCreateOptionsMenu( Menu menu ) 
    {
       MenuInflater inflater = getMenuInflater();
       inflater.inflate( R.menu.splashmenu, menu );
       return true;
    }
    
     /** User Option selection menu
     */
    @Override
    public boolean onOptionsItemSelected( MenuItem item )
    {
       // Handle item selection
       switch( item.getItemId() ) 
       {
           case R.id.main_start:
               Intent startActivity = new Intent(getBaseContext(), FastCVSample.class);
               startActivity( startActivity );

               return true;

           case R.id.settings:
               Intent settingsActivity = new Intent( getBaseContext(), Preferences.class );
               startActivity( settingsActivity );

               return true;
           default:
              return super.onOptionsItemSelected(item);
       }
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


    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "onResuestPermissionResult " + grantResults[0]);
        } else{
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    //Native Function Declarations
    public native String getFastCVVersion();
}
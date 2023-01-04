package project.test.fastcv_app;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import project.test.fastcv_app.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'fastcv_app' library on application startup.
    static {
        System.loadLibrary("fastcv_app");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(getFastCVVersion());
    }

    /**
     * A native method that is implemented by the 'fastcv_app' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native String getFastCVVersion();
}
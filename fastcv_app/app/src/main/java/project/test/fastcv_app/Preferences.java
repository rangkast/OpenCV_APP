/*==============================================================================
            Copyright (c) 2010-2011 Qualcomm Technologies Incorporated.
            All Rights Reserved.
            Qualcomm Technologies Confidential and Proprietary
==============================================================================*/

package project.test.fastcv_app;
 
import android.os.Bundle;
import android.preference.PreferenceActivity;

 
/**
 * Basic empty Preferences class. All configurations for 
 * Setting menu is done in the xml file. 
 * 
 */
public class Preferences extends PreferenceActivity 
{
   /** Called when the activity is first created. */
   @Override
   protected void onCreate( Bundle savedInstanceState ) 
   {
      super.onCreate(savedInstanceState);
      addPreferencesFromResource( R.xml.preferences );
   }
}


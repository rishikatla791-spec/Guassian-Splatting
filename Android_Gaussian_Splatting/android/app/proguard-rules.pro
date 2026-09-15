# ---------------------------------------------------------------------------
# JNI boundary -- names here are resolved as STRINGS at runtime, so anything
# R8 renames simply fails to link, at the moment training starts.
# ---------------------------------------------------------------------------

# `external fun` declarations bind by the mangled symbol
# Java_com_splat_mobile3dgs_engine_NativeBrushEngine_nativeXxx, which encodes the
# package, class and method name. Renaming any of them breaks the lookup.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}
-keep class com.splat.mobile3dgs.engine.NativeBrushEngine { *; }

# brush_bridge.cpp resolves these two by name via GetMethodID on whatever object
# it is handed, so the interface AND every implementation must keep them.
-keep interface com.splat.mobile3dgs.engine.TrainingProgressListener { *; }
-keep class * implements com.splat.mobile3dgs.engine.TrainingProgressListener {
    void onProgress(int, float);
    void onState(java.lang.String, java.lang.String);
}

# Values crossing the JNI boundary as whole objects.
-keep class com.splat.mobile3dgs.engine.TrainingResult { *; }
-keep class com.splat.mobile3dgs.engine.SplatValidation { *; }

# ---------------------------------------------------------------------------
# WebView viewer -- the splat renderer is JavaScript calling into an injected
# Kotlin object by name.
# ---------------------------------------------------------------------------
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Keep source line numbers so a release crash report is still readable.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

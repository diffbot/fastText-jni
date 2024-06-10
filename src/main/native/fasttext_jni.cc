#include <strstream>
#include <sstream>

#include "fasttext_jni.h"

#include "fastText/src/fasttext.h"

class FastTextWrapper : public fasttext::FastText {
public:
    void loadModelFromFileStream(std::istream& in) {
        if (!this->checkModel(in)) {
            throw std::invalid_argument("Invalid model format");
        }
        this->loadModel(in);
    }
};

// Cache Prediction class and constructor
static jmethodID predictionConstructor;
static jclass predictionClass;

// According to http://docs.oracle.com/javase/1.5.0/docs/guide/jni/spec/invocation.html#JNI_OnLoad
// The VM calls JNI_OnLoad when the native library is loaded (for example, through System.loadLibrary).
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    } else {
        jclass localSimpleCls = env->FindClass("com/diffbot/fasttext/Prediction");

        if (localSimpleCls == NULL) {
            return JNI_ERR;
        }
        predictionClass = (jclass) env->NewGlobalRef(localSimpleCls);
        predictionConstructor = env->GetMethodID(predictionClass, "<init>", "(FLjava/lang/String;)V");
    }
    return JNI_VERSION_1_6;
}

// According to http://docs.oracle.com/javase/1.5.0/docs/guide/jni/spec/invocation.html#JNI_OnUnload
// The VM calls JNI_OnUnload when the class loader containing the native library is garbage collected.
void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        // Something is wrong but nothing we can do about this :(
        return;
    } else {
        env->DeleteGlobalRef(predictionClass);
    }
}


static jfieldID handleFieldID = NULL;

JNIEXPORT void JNICALL Java_com_diffbot_fasttext_FastTextModel_load
(JNIEnv *env, jobject obj, jobject bytebuffer) {
    jlong buffsize = env->GetDirectBufferCapacity(bytebuffer);
    jbyte *buff = (jbyte *) env->GetDirectBufferAddress(bytebuffer);
    FastTextWrapper *ft = new FastTextWrapper;
    std::istrstream stream(reinterpret_cast<const char*>(buff), buffsize);
    ft->loadModelFromFileStream(stream);

    if (handleFieldID == NULL) {
        jclass c = env->GetObjectClass(obj);
        handleFieldID = env->GetFieldID(c, "handle", "J");
    }
    env->SetLongField(obj, handleFieldID, (jlong) ft);
}

JNIEXPORT jobject JNICALL Java_com_diffbot_fasttext_FastTextModel_predictProba
(JNIEnv *env, jobject obj, jstring s) {
    jboolean isCopy;
    const char* utf_string = env->GetStringUTFChars(s, &isCopy);
    FastTextWrapper *ft = (FastTextWrapper *) env->GetLongField(obj, handleFieldID);
    std::vector<std::pair<fasttext::real,std::string>> predictions;
    std::string text = utf_string;
    std::stringstream stream;
    stream << text << '\n';
    bool result = ft->predictLine(stream, predictions, 1, 0.0);
    env->ReleaseStringUTFChars(s, utf_string);
    if (!result || predictions.size() == 0) {
        return NULL;
    }
    std::pair<fasttext::real, std::string> prediction = predictions[0];
    jstring label = env->NewStringUTF(prediction.second.c_str());
    return env->NewObject(predictionClass, predictionConstructor, prediction.first, label);
}

JNIEXPORT jobjectArray JNICALL Java_com_diffbot_fasttext_FastTextModel_predictProbaTopK
(JNIEnv *env, jobject obj, jstring s, jint k) {

    jboolean isCopy;
    const char* utf_string = env->GetStringUTFChars(s, &isCopy);
    FastTextWrapper *ft = (FastTextWrapper *) env->GetLongField(obj, handleFieldID);
    std::vector<std::pair<fasttext::real,std::string>> predictions;
    std::string text = utf_string;
    std::stringstream stream;
    stream << text << '\n';
    bool result = ft->predictLine(stream, predictions, k, -1);
    env->ReleaseStringUTFChars(s, utf_string);
    if (!result || predictions.size() == 0) {
        return NULL;
    }

    k = predictions.size();
    jobjectArray top_k_predictions = env->NewObjectArray(k, predictionClass, nullptr);

    for (int i = 0; i < k; i++) {
        std::pair<fasttext::real, std::string> prediction = predictions[i];

        jstring label = env->NewStringUTF(prediction.second.c_str());
        jobject pred = env->NewObject(predictionClass, predictionConstructor, prediction.first, label);
        env->SetObjectArrayElement(top_k_predictions, i, pred);
    }

    return top_k_predictions;
}

JNIEXPORT jobjectArray JNICALL Java_com_diffbot_fasttext_FastTextModel_predictProbaWithThreshold
(JNIEnv *env, jobject obj, jstring s, jfloat threshold) {

    jboolean isCopy;
    const char* utf_string = env->GetStringUTFChars(s, &isCopy);
    FastTextWrapper *ft = (FastTextWrapper *) env->GetLongField(obj, handleFieldID);
    std::vector<std::pair<fasttext::real,std::string>> predictions;
    std::string text = utf_string;
    std::stringstream stream;
    stream << text << '\n';
    bool result = ft->predictLine(stream, predictions, -1, threshold);
    env->ReleaseStringUTFChars(s, utf_string);
    if (!result || predictions.size() == 0) {
        return NULL;
    }

    jsize k = predictions.size();
    jobjectArray ret = env->NewObjectArray(k, predictionClass, nullptr);

    for (int i = 0; i < k; i++) {
        std::pair<fasttext::real, std::string> prediction = predictions[i];

        jstring label = env->NewStringUTF(prediction.second.c_str());
        jobject pred = env->NewObject(predictionClass, predictionConstructor, prediction.first, label);
        env->SetObjectArrayElement(ret, i, pred);
    }

    return ret;
}

JNIEXPORT void JNICALL Java_com_diffbot_fasttext_FastTextModel_close
(JNIEnv *env, jobject obj) {
    FastTextWrapper *ft = (FastTextWrapper *) env->GetLongField(obj, handleFieldID);
    delete ft;
}

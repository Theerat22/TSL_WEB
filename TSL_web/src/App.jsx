import React, { useState, useRef, useEffect } from 'react';
import { Camera, Video, Square, Home, Upload, Loader } from 'lucide-react';

export default function ThaiSignTranslation() {
  const [isRecording, setIsRecording] = useState(false);
  const [translationResult, setTranslationResult] = useState('Translation will appear here...');
  const [activeTab, setActiveTab] = useState('translation');
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [flaskUrl] = useState('http://localhost:5001'); // URL ของ Flask server
  
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  // เริ่มกล้อง
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 1280, 
          height: 720,
          frameRate: 30 
        }, 
        audio: false 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setTranslationResult('Error: Cannot access camera. Please check permissions.');
    }
  };

  // หยุดกล้อง
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  // เริ่มนับเวลา
  const startTimer = () => {
    setRecordingTime(0);
    timerRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  // หยุดนับเวลา
  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // แปลงวินาทีเป็นรูปแบบ mm:ss
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // เริ่ม/หยุดการบันทึก
  const handleRecording = async () => {
    if (!isRecording) {
      // เริ่มบันทึก
      try {
        if (!streamRef.current) {
          await startCamera();
        }

        const mediaRecorder = new MediaRecorder(streamRef.current, {
          mimeType: 'video/webm;codecs=vp9'
        });
        
        const chunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          setRecordedBlob(blob);
          console.log('Recording completed, blob size:', blob.size);
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start();
        
        setIsRecording(true);
        setTranslationResult('Recording... Please perform your sign language gesture.');
        startTimer();
        
      } catch (err) {
        console.error('Error starting recording:', err);
        setTranslationResult('Error starting recording: ' + err.message);
      }
    } else {
      // หยุดบันทึก
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        stopTimer();
        setTranslationResult('Recording completed. Processing...');
      }
    }
  };

  // ส่งวิดีโอไปยัง Flask server
  const sendVideoToFlask = async (videoBlob) => {
    if (!videoBlob) {
      setTranslationResult('Error: No video recorded');
      return;
    }

    setIsProcessing(true);
    setTranslationResult('Sending video to AI model for translation...');

    try {
      const formData = new FormData();
      formData.append('video', videoBlob, 'recording.webm');

      const response = await fetch(`${flaskUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // แสดงผลการแปล
        const thaiText = result.thai_text || 'ไม่สามารถแปลได้';
        const englishText = result.english_text || 'Translation not available';
        const confidence = result.confidence || 0;
        
        setTranslationResult(`${thaiText} - ${englishText}\n(Confidence: ${(confidence * 100).toFixed(1)}%)`);
      } else {
        setTranslationResult(`Translation failed: ${result.error || 'Unknown error'}`);
      }

    } catch (error) {
      console.error('Error sending video to Flask:', error);
      setTranslationResult(`Connection error: ${error.message}. Please check if Flask server is running at ${flaskUrl}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // อัปโหลดวิดีโอด้วยตนเอง
  const handleManualUpload = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setRecordedBlob(file);
        setTranslationResult('Video file selected. Click "Send to AI" to translate.');
      }
    };
    input.click();
  };

  // เมื่อมีการบันทึกเสร็จ ให้ส่งไปยัง Flask อัตโนมัติ
  useEffect(() => {
    if (recordedBlob && !isProcessing) {
      sendVideoToFlask(recordedBlob);
    }
  }, [recordedBlob]);

  // เริ่มกล้องเมื่อโหลดหน้า
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
      stopTimer();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            {/* Logo/Brand */}
            <div className="flex items-center space-x-3">
              <div className="text-xl font-semibold text-gray-900">
                Chitralada School
              </div>
            </div>
            
            {/* Navigation Pills */}
            <nav className="flex space-x-1">
              <button
                onClick={() => setActiveTab('home')}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeTab === 'home'
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Home className="w-4 h-4 inline mr-1" />
                Home
              </button>
              <button
                onClick={() => setActiveTab('translation')}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeTab === 'translation'
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                Translation
              </button>
            </nav>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Title */}
        <div className="text-center mb-8 mt-4">
          <h1 className="text-3xl md:text-4xl font-bold text-blue-600 mb-2">
            AI-powered Thai Sign Language Translation
          </h1>
          <h2 className="text-xl md:text-2xl font-semibold text-blue-500">
            Web Application
          </h2>
        </div>

        {/* Flask Server Settings */}
        {/* <div className="max-w-md mx-auto mb-6">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Flask Server URL:
            </label>
            <input
              type="text"
              value={flaskUrl}
              onChange={(e) => setFlaskUrl(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="http://localhost:5000"
            />
          </div>
        </div> */}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Video Recording Section */}
          <div className="space-y-6">
            {/* Video Box */}
            <div className="relative">
              <div className="w-full h-64 sm:h-80 lg:h-96 bg-black rounded-lg overflow-hidden border-2 border-gray-300">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
                
                {/* No camera placeholder */}
                {!streamRef.current && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
                    <div className="text-center text-gray-400">
                      <Camera className="w-16 h-16 mx-auto mb-4" />
                      <span className="text-lg">Camera Loading...</span>
                      <p className="text-sm mt-2">Please allow camera access</p>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Recording Status Indicator */}
              {isRecording && (
                <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                  REC {formatTime(recordingTime)}
                </div>
              )}

              {/* Processing Indicator */}
              {isProcessing && (
                <div className="absolute top-4 left-4 bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center">
                  <Loader className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </div>
              )}
            </div>

            {/* Control Buttons */}
            <div className="text-center space-y-4">
              {/* Record Button */}
              <button
                onClick={handleRecording}
                disabled={isProcessing}
                className={`inline-flex items-center px-8 py-3 rounded-full font-semibold text-lg transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed ${
                  isRecording
                    ? 'bg-gray-600 hover:bg-gray-700 text-white'
                    : 'bg-red-500 hover:bg-red-600 text-white shadow-lg hover:shadow-xl'
                }`}
              >
                {isRecording ? (
                  <>
                    <Square className="w-5 h-5 mr-2 fill-current" />
                    Stop Recording
                  </>
                ) : (
                  <>
                    <Video className="w-5 h-5 mr-2" />
                    Start Recording
                  </>
                )}
              </button>

              {/* Manual Upload Button */}
              <div className="flex justify-center space-x-4">
                <button
                  onClick={handleManualUpload}
                  disabled={isProcessing}
                  className="inline-flex items-center px-6 py-2 rounded-full font-medium text-sm transition-colors border border-blue-500 text-blue-500 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Video File
                </button>

                {/* Send to AI Button */}
                {recordedBlob && (
                  <button
                    onClick={() => sendVideoToFlask(recordedBlob)}
                    disabled={isProcessing}
                    className="inline-flex items-center px-6 py-2 rounded-full font-medium text-sm transition-colors bg-green-500 hover:bg-green-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <>
                        <Loader className="w-4 h-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      'Send to AI'
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Translation Result Section */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              Translation Result
            </h3>
            
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
              <div className="p-6">
                <div className="min-h-32 lg:min-h-48">
                  <p className={`text-xl leading-relaxed whitespace-pre-line ${
                    translationResult.includes('Translation will appear') || 
                    translationResult.includes('Recording...') ||
                    translationResult.includes('Processing...') ||
                    translationResult.includes('Sending...')
                      ? 'text-gray-500 italic' 
                      : 'text-gray-800'
                  }`}>
                    {translationResult}
                  </p>
                </div>
              </div>
            </div>

            {/* Recording Status */}
            {isRecording && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center text-red-700">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse mr-3"></div>
                  <span className="font-medium">Recording in progress...</span>
                </div>
                <div className="mt-2 text-sm text-red-600">
                  Duration: {formatTime(recordingTime)}
                </div>
              </div>
            )}

            {/* Processing Status */}
            {isProcessing && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center text-blue-700">
                  <Loader className="w-4 h-4 animate-spin mr-3" />
                  <span className="font-medium">AI is analyzing your sign language...</span>
                </div>
              </div>
            )}

            {/* Video Ready Status */}
            {recordedBlob && !isProcessing && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center text-green-700">
                  <Video className="w-4 h-4 mr-3" />
                  <span className="font-medium">Video ready for processing</span>
                </div>
                <div className="mt-2 text-sm text-green-600">
                  Size: {(recordedBlob.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Instructions Section */}
        <div className="mt-12 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">1</span>
              </div>
              <h4 className="font-semibold mb-2">Setup Camera</h4>
              <p className="text-sm text-gray-600">
                Allow camera access and position yourself clearly in frame
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">2</span>
              </div>
              <h4 className="font-semibold mb-2">Record Signs</h4>
              <p className="text-sm text-gray-600">
                Click record and perform your Thai sign language gestures
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">3</span>
              </div>
              <h4 className="font-semibold mb-2">Stop Recording</h4>
              <p className="text-sm text-gray-600">
                Stop recording when finished - video will be sent automatically
              </p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold text-xl">4</span>
              </div>
              <h4 className="font-semibold mb-2">Get Translation</h4>
              <p className="text-sm text-gray-600">
                AI will analyze and display Thai and English translation
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
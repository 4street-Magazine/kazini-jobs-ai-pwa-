// Firebase background push for Chania AI. Keep this file in the same folder as index.html (the site root).
// Use the same values as FIREBASE_CONFIG in index.html.
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: 'AIzaSyALr2O4ydif18Er5MOGCEzxPhfOLsDYe84',
  authDomain: 'chania-ai-22392.firebaseapp.com',
  projectId: 'chania-ai-22392',
  messagingSenderId: '647948327540',
  appId: '1:647948327540:web:74fafdb834aa07d00a0f21',
});

// Messages that carry a notification are shown by Firebase itself; tapping one opens the link the worker set (APP_URL).
firebase.messaging();

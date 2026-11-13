// Chania AI service worker: always fetch the latest page when online, cache for offline use.
// Bump VERSION when you want to clear all old cached assets (icons, images).
const VERSION = 'chania-v3';
const CORE = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png', './job-match.mp3'];
const STATIC_FILE = /\.(png|jpe?g|svg|webp|ico|gif|mp3|woff2?)$/i;   // only these are cached; API paths such as /matches or /insights are never touched, even on the same address

self.addEventListener('install', e => {
  // Pre-cache the core files so the app opens offline; one missing file must not block installation
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    await Promise.all(CORE.map(u => c.add(u).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;           // never touch other sites

  // Pages and manifest: network first, so every update shows immediately
  if (req.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html') || url.pathname.endsWith('manifest.json')) {
    e.respondWith(
      fetch(req, { cache: 'no-store' })
        .then(res => { if (res.ok) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); } return res; })
        .catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
    return;
  }

  // Images, icons and sounds: serve cached, refresh in the background. Anything else (the worker API) goes straight to the network.
  if (!STATIC_FILE.test(url.pathname)) return;
  e.respondWith(
    caches.match(req).then(cached => {
      const net = fetch(req).then(res => {
        if (res.status === 200) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
        return res;
      }).catch(() => cached);
      return cached || net;
    })
  );
});

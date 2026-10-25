// Chania AI worker: career profiles -> D1, Bright Data jobs -> D1, embeddings -> Vectorize, semantic matching, SMS alerts.

const EMBED_MODEL = '@cf/baai/bge-base-en-v1.5'; // 768 dimensions
const BATCH = 50;

/* ---------- helpers ---------- */
const cors = env => ({
  'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
});
const json = (env, data, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json', ...cors(env) } });

async function sha(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
}

async function embed(env, texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const res = await env.AI.run(EMBED_MODEL, { text: texts.slice(i, i + BATCH) });
    out.push(...res.data);
  }
  return out;
}

const clean = s => String(s ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();

/* ---------- profile -> text ---------- */
function profileToText(p) {
  const parts = [];
  const add = (label, v) => { const t = clean(Array.isArray(v) ? v.map(x => typeof x === 'object' ? Object.values(x).join(' ') : x).join(', ') : v); if (t) parts.push(`${label}: ${t}`); };
  add('Job title', p.job_title); add('Education', p.education); add('Course', p.course);
  add('Institution', p.institution); add('Skills', p.skills); add('Experience years', p.years_experience);
  add('Current workplace', p.workplace); add('Responsibilities', p.responsibilities);
  add('Past jobs', p.past_jobs || p.pastJobs); add('Certifications', p.certifications);
  add('Projects', p.project_details); add('Achievements', p.achievements); add('Languages', p.languages);
  return parts.join('\n').slice(0, 2500);
}

function normalisePhone(raw) {
  const d = String(raw || '').replace(/\D/g, '');
  if (d.startsWith('254') && d.length === 12) return '+' + d;
  if (d.length === 10 && d.startsWith('0')) return '+254' + d.slice(1);
  if (d.length === 9) return '+254' + d;
  return '';
}

/* ---------- POST /profile ---------- */
async function saveProfile(env, ctx, body) {
  const { deviceId, status, profile } = body || {};
  if (!deviceId || !profile || typeof profile !== 'object') return { error: 'deviceId and profile required', code: 400 };

  const text = profileToText(profile);
  const hash = await sha(text);
  const phone = normalisePhone(profile.phone);

  await env.DB.prepare(`
    INSERT INTO profiles (device_id, full_name, phone, data, status, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
    ON CONFLICT(device_id) DO UPDATE SET
      full_name = ?2, phone = ?3, data = ?4, status = ?5, updated_at = datetime('now')
  `).bind(deviceId, clean(profile.full_name), phone, JSON.stringify(profile), status === 'submitted' ? 'submitted' : 'draft').run();

  const row = await env.DB.prepare('SELECT id, text_hash, embedded FROM profiles WHERE device_id = ?').bind(deviceId).first();

  // Only embed finished profiles whose content changed
  if (status === 'submitted' && text && (row.text_hash !== hash || !row.embedded)) {
    ctx.waitUntil(embedAndMatchProfile(env, row.id, text, hash));
  }
  return { ok: true, profileId: row.id };
}

async function embedAndMatchProfile(env, profileId, text, hash) {
  const [vec] = await embed(env, [text]);
  await env.PROFILE_VEC.upsert([{ id: `p_${profileId}`, values: vec, metadata: { profileId } }]);
  await env.DB.prepare('UPDATE profiles SET profile_text = ?, text_hash = ?, embedded = 1 WHERE id = ?').bind(text, hash, profileId).run();

  // Backfill: match this profile against jobs already in the database
  const res = await env.JOB_VEC.query(vec, { topK: 20, returnMetadata: 'none' });
  const min = parseFloat(env.MATCH_THRESHOLD || '0.62');
  const stmts = res.matches.filter(m => m.score >= min).map(m =>
    env.DB.prepare('INSERT OR IGNORE INTO matches (profile_id, job_id, score) VALUES (?, ?, ?)').bind(profileId, m.id.replace(/^j_/, ''), m.score));
  if (stmts.length) await env.DB.batch(stmts);
  await notifyPending(env);
}

/* ---------- Bright Data: trigger + webhook ---------- */
async function triggerBrightData(env) {
  if (!env.BRIGHTDATA_API_KEY || !env.BRIGHTDATA_DATASET_ID) return { skipped: 'Bright Data not configured' };
  const params = new URLSearchParams({
    dataset_id: env.BRIGHTDATA_DATASET_ID,
    format: 'json',
    type: 'discover_new',
    discover_by: 'keyword',
    endpoint: `${env.WORKER_URL}/webhook/brightdata`,
    uncompressed_webhook: 'true',
    webhook_header_Authorization: `Bearer ${env.WEBHOOK_SECRET}`,
  });
  const res = await fetch(`https://api.brightdata.com/datasets/v3/trigger?${params}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.BRIGHTDATA_API_KEY}`, 'Content-Type': 'application/json' },
    body: env.BRIGHTDATA_INPUTS || '[]',
  });
  return { status: res.status, body: await res.text() };
}

const pick = (o, ...keys) => { for (const k of keys) { const v = o?.[k]; if (v) return clean(v); } return ''; };

const normaliseType = t => /part/i.test(t) ? 'Part-Time' : /full/i.test(t) ? 'Full-Time' : t;

function mapJob(r) {
  return {
    title: pick(r, 'job_title', 'title'),
    company: pick(r, 'company_name', 'company'),
    location: pick(r, 'job_location', 'location'),
    description: pick(r, 'job_summary', 'job_description_formatted', 'job_description', 'description'),
    apply_url: pick(r, 'apply_link', 'url', 'job_url'),
    posted_at: pick(r, 'job_posted_date', 'job_posted_time', 'posted_at'),
    employment_type: normaliseType(pick(r, 'job_employment_type', 'employment_type', 'job_type')),
    remote: /remote/i.test([pick(r, 'job_work_type', 'work_type', 'workplace_type'), pick(r, 'job_location', 'location'), pick(r, 'job_title', 'title')].join(' ')) ? 1 : 0,
  };
}

function parseRecords(text) {
  try { const d = JSON.parse(text); return Array.isArray(d) ? d : [d]; }
  catch { return text.split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean); } // NDJSON
}

async function ingestJobs(env, records) {
  const jobs = [];
  for (const r of records) {
    const j = mapJob(r);
    if (!j.title) continue;
    j.id = (await sha(j.apply_url || `${j.title}|${j.company}|${j.location}`)).slice(0, 32);
    jobs.push(j);
  }
  // Insert; meta.changes === 1 means the job is new (duplicates are ignored)
  const results = await env.DB.batch(jobs.map(j =>
    env.DB.prepare('INSERT OR IGNORE INTO jobs (id, title, company, location, description, apply_url, posted_at, employment_type, remote) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
      .bind(j.id, j.title, j.company, j.location, j.description, j.apply_url, j.posted_at, j.employment_type, j.remote)));
  const fresh = jobs.filter((_, i) => results[i].meta.changes === 1);
  await embedAndMatchJobs(env, fresh);
}

const jobToText = j => `${j.title}\n${j.company}\n${j.location}\n${j.description || ''}`.slice(0, 2000);

async function embedAndMatchJobs(env, jobs) {
  if (!jobs.length) return;
  const vecs = await embed(env, jobs.map(jobToText));
  await env.JOB_VEC.upsert(jobs.map((j, i) => ({ id: `j_${j.id}`, values: vecs[i], metadata: { title: j.title.slice(0, 100) } })));
  await env.DB.batch(jobs.map(j => env.DB.prepare('UPDATE jobs SET embedded = 1 WHERE id = ?').bind(j.id)));

  // Match every new job against all embedded profiles
  const min = parseFloat(env.MATCH_THRESHOLD || '0.62');
  const stmts = [];
  for (let i = 0; i < jobs.length; i++) {
    const res = await env.PROFILE_VEC.query(vecs[i], { topK: 50, returnMetadata: 'none' });
    for (const m of res.matches) {
      if (m.score < min) continue;
      stmts.push(env.DB.prepare('INSERT OR IGNORE INTO matches (profile_id, job_id, score) VALUES (?, ?, ?)').bind(parseInt(m.id.replace(/^p_/, ''), 10), jobs[i].id, m.score));
    }
  }
  for (let i = 0; i < stmts.length; i += 90) await env.DB.batch(stmts.slice(i, i + 90));
  await notifyPending(env);
}

// Retry jobs whose embedding failed earlier
async function retryUnembedded(env) {
  const { results } = await env.DB.prepare('SELECT * FROM jobs WHERE embedded = 0 LIMIT 100').all();
  await embedAndMatchJobs(env, results);
}

/* ---------- SMS alerts (Africa's Talking; swap for your provider) ---------- */
async function sendSms(env, to, message) {
  if (!env.AT_API_KEY || !env.AT_USERNAME) { console.log('SMS (not configured):', to, message); return false; }
  const res = await fetch('https://api.africastalking.com/version1/messaging', {
    method: 'POST',
    headers: { apiKey: env.AT_API_KEY, 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
    body: new URLSearchParams({ username: env.AT_USERNAME, to, message }),
  });
  return res.ok;
}

// One SMS per profile summarising its unnotified matches
async function notifyPending(env) {
  const { results } = await env.DB.prepare(`
    SELECT p.id AS pid, p.phone, p.full_name, COUNT(*) AS n,
           (SELECT j.title || ' at ' || COALESCE(j.company, '') FROM matches m2 JOIN jobs j ON j.id = m2.job_id
             WHERE m2.profile_id = p.id AND m2.notified = 0 ORDER BY m2.score DESC LIMIT 1) AS top
    FROM matches m JOIN profiles p ON p.id = m.profile_id
    WHERE m.notified = 0 AND p.phone != '' GROUP BY p.id LIMIT 200`).all();
  for (const r of results) {
    const first = (r.full_name || '').split(' ')[0] || 'there';
    const msg = r.n === 1
      ? `Hi ${first}, Chania AI found a job matching your profile: ${r.top}. Open Chania AI to apply.`
      : `Hi ${first}, Chania AI found ${r.n} jobs matching your profile, including ${r.top}. Open Chania AI to apply.`;
    if (await sendSms(env, r.phone, msg)) await env.DB.prepare('UPDATE matches SET notified = 1 WHERE profile_id = ?').bind(r.pid).run();
  }
}

/* ---------- entry points ---------- */
export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (req.method === 'OPTIONS') return new Response(null, { headers: cors(env) });
    try {
      if (url.pathname === '/health') return json(env, { ok: true });

      if (url.pathname === '/profile' && req.method === 'POST') {
        const r = await saveProfile(env, ctx, await req.json());
        return json(env, r, r.code || 200);
      }

      // The signed-in user's matched jobs only (the app filters Remote / Full-Time / Part-Time on this list)
      if (url.pathname === '/matches' && req.method === 'GET') {
        const deviceId = url.searchParams.get('deviceId');
        if (!deviceId) return json(env, { error: 'deviceId required' }, 400);
        const { results } = await env.DB.prepare(`
          SELECT j.id, j.title, j.company, j.location, j.description, j.apply_url, j.posted_at, j.employment_type, j.remote, m.score
          FROM matches m JOIN profiles p ON p.id = m.profile_id JOIN jobs j ON j.id = m.job_id
          WHERE p.device_id = ? ORDER BY m.score DESC LIMIT 100`).bind(deviceId).all();
        return json(env, { matches: results });
      }

      if (url.pathname === '/webhook/brightdata' && req.method === 'POST') {
        if (req.headers.get('Authorization') !== `Bearer ${env.WEBHOOK_SECRET}`) return json(env, { error: 'unauthorized' }, 401);
        const records = parseRecords(await req.text());
        ctx.waitUntil(ingestJobs(env, records));   // reply fast; Bright Data expects 200 within 30s
        return json(env, { received: records.length });
      }

      if (url.pathname === '/admin/trigger' && req.method === 'POST') {
        if (req.headers.get('Authorization') !== `Bearer ${env.ADMIN_SECRET}`) return json(env, { error: 'unauthorized' }, 401);
        return json(env, await triggerBrightData(env));
      }
      return json(env, { error: 'not found' }, 404);
    } catch (e) {
      console.error(e);
      return json(env, { error: 'server error' }, 500);
    }
  },

  async scheduled(_evt, env, ctx) {
    ctx.waitUntil((async () => { await retryUnembedded(env); await triggerBrightData(env); })());
  },
};

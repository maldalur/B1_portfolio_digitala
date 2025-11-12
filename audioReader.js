/* Unified Audio Reader - Euskara / Español / English / Français */
(function(window){
  const TARGET_LANGS = ["es-ES","eu-ES","en-US","fr-FR"];
  let voices = [];
  let isReading = false;
  let isPaused = false;
  let utterance = null;
  let queue = [];
  let currentElement = null;
  let contentElements = [];
  let selectedLanguage = localStorage.getItem('audioReaderLang') || 'es-ES';
  let index = 0; // index inside contentElements
  let chunkIndex = 0; // index inside chunks of current element
  let elementChunks = [];

  function log(...args){ console.log('[AudioReader]', ...args); }

  function loadVoices(){
    voices = window.speechSynthesis.getVoices();
    if(!voices || voices.length === 0){
      return setTimeout(loadVoices, 150); // retry until available
    }
    buildLanguageOptions();
  }

  function buildLanguageOptions(){
    const select = document.getElementById('languageSelect');
    if(!select) return;
    select.innerHTML = '';
    TARGET_LANGS.forEach(lang => {
      const opt = document.createElement('option');
      opt.value = lang;
      opt.textContent = labelForLang(lang);
      const available = voices.some(v => v.lang === lang || v.lang.startsWith(lang.substring(0,2)));
      if(!available){
        opt.disabled = true;
        opt.textContent += ' (ez dago / no disponible)';
      }
      select.appendChild(opt);
    });
    // restore selection
    if(TARGET_LANGS.includes(selectedLanguage)) select.value = selectedLanguage;
  }

  function labelForLang(lang){
    switch(lang){
      case 'es-ES': return '🇪🇸 Español';
      case 'eu-ES': return '🇪🇺 Euskara';
      case 'en-US': return '🇬🇧 English';
      case 'fr-FR': return '🇫🇷 Français';
      default: return lang;
    }
  }

  function init(opts={}){
    const contentSelector = opts.contentSelector || '.content';
    const contentRoot = document.querySelector(contentSelector);
    if(!contentRoot){ log('Ez da aurkitu edukia:', contentSelector); return; }
    contentElements = Array.from(contentRoot.querySelectorAll('h1,h2,h3,h4,p,li'))
      .filter(el => {
        const text = (el.textContent||'').trim();
        return text.length > 0 && !el.querySelector('img');
      });

    attachPanel();
    // load voices
    loadVoices();
    if (speechSynthesis.onvoiceschanged !== undefined){
      speechSynthesis.onvoiceschanged = loadVoices;
    }
    log('Prest! Edukiak:', contentElements.length);
  }

  function attachPanel(){
    if(document.getElementById('audioReaderPanel')) return; // already there
    const panel = document.createElement('div');
    panel.id = 'audioReaderPanel';
    panel.className = 'audio-reader-panel';
    panel.style.cssText = 'position:fixed;bottom:20px;left:20px;z-index:1000;background:#fff;border-radius:15px;padding:15px;box-shadow:0 4px 15px rgba(0,0,0,0.2);display:none;max-width:260px;font-family:inherit;';
    panel.innerHTML = `<div style="margin-bottom:10px;">
        <label for="languageSelect" style="display:block;margin-bottom:5px;font-weight:600;color:#667eea;">🌐 Hizkuntza / Idioma</label>
        <select id="languageSelect" style="width:100%;padding:8px;border:2px solid #667eea;border-radius:8px;font-size:14px;cursor:pointer;"></select>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
        <button id="playPauseBtn" aria-label="Play/Pause" style="flex:1 0 70px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:8px;padding:10px 12px;font-size:14px;cursor:pointer;font-weight:600;">▶️ Play</button>
        <button id="stopBtn" aria-label="Stop" style="flex:1 0 70px;background:#ff9800;color:#fff;border:none;border-radius:8px;padding:10px 12px;font-size:14px;cursor:pointer;font-weight:600;">⏹️ Stop</button>
        <button id="closeAudioPanel" aria-label="Close" style="flex:1 0 70px;background:#dc3545;color:#fff;border:none;border-radius:8px;padding:10px 12px;font-size:14px;cursor:pointer;font-weight:600;">❌ Itxi</button>
      </div>`;
    document.body.appendChild(panel);

    const openBtn = document.createElement('button');
    openBtn.id = 'audioReaderBtn';
    openBtn.setAttribute('aria-label','Ireki irakurketa panela');
    openBtn.style.cssText = 'position:fixed;bottom:20px;left:20px;z-index:999;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:50%;width:60px;height:60px;font-size:26px;cursor:pointer;box-shadow:0 4px 10px rgba(0,0,0,0.3);transition:.3s;';
    openBtn.textContent = '🔊';
    document.body.appendChild(openBtn);

    openBtn.addEventListener('click', ()=>{
      panel.style.display='block';
      openBtn.style.display='none';
      document.getElementById('languageSelect').focus();
    });
    document.getElementById('closeAudioPanel').addEventListener('click', ()=>{
      stop();
      panel.style.display='none';
      openBtn.style.display='block';
      openBtn.focus();
    });
    document.getElementById('languageSelect').addEventListener('change', e => {
      selectedLanguage = e.target.value;
      localStorage.setItem('audioReaderLang', selectedLanguage);
      if(isReading){
        // restart current element in new language
        speechSynthesis.cancel();
        chunkIndex = 0; queue = [];
        speakCurrentElement();
      }
    });
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
    document.getElementById('stopBtn').addEventListener('click', stop);

    // keyboard shortcuts
    window.addEventListener('keydown', (e)=>{
      if(e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
      if(e.code === 'Space'){ e.preventDefault(); togglePlayPause(); }
      if(e.key.toLowerCase() === 's'){ stop(); }
    });
  }

  function togglePlayPause(){
    const btn = document.getElementById('playPauseBtn');
    if(!isReading){
      start();
      btn.textContent = '⏸️ Pause';
      btn.style.background = 'linear-gradient(135deg,#f093fb 0%,#f5576c 100%)';
    } else if(isPaused){
      window.speechSynthesis.resume();
      isPaused = false;
      btn.textContent = '⏸️ Pause';
      btn.style.background = 'linear-gradient(135deg,#f093fb 0%,#f5576c 100%)';
    } else {
      window.speechSynthesis.pause();
      isPaused = true;
      btn.textContent = '▶️ Play';
      btn.style.background = 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)';
    }
  }

  function start(){
    if(contentElements.length === 0){ log('Ez dago edukirik irakurtzeko'); return; }
    isReading = true; isPaused = false; index = 0; chunkIndex = 0; queue = [];
    speakCurrentElement();
  }

  function stop(){
    window.speechSynthesis.cancel();
    isReading = false; isPaused = false; index = 0; chunkIndex = 0; queue = []; elementChunks = [];
    clearHighlight();
    const btn = document.getElementById('playPauseBtn');
    if(btn){ btn.textContent='▶️ Play'; btn.style.background='linear-gradient(135deg,#667eea 0%,#764ba2 100%)'; }
  }

  function speakCurrentElement(){
    if(index >= contentElements.length){ stop(); return; }
    currentElement = contentElements[index];
    highlight(currentElement);
    const text = (currentElement.textContent||'').trim();
    elementChunks = chunkText(text, 350); // 350 char chunks
    chunkIndex = 0;
    speakNextChunk();
  }

  function speakNextChunk(){
    if(chunkIndex >= elementChunks.length){
      clearHighlight(currentElement);
      index++; chunkIndex = 0; elementChunks = [];
      if(isReading){ setTimeout(speakCurrentElement, 120); }
      return;
    }
    const chunk = elementChunks[chunkIndex];
    utterance = new SpeechSynthesisUtterance(chunk);
    utterance.lang = selectedLanguage;
    utterance.rate = 0.9; utterance.pitch = 1; utterance.volume = 1;
    const voice = pickBestVoice(selectedLanguage);
    if(voice) utterance.voice = voice;
    utterance.onend = ()=>{ chunkIndex++; speakNextChunk(); };
    utterance.onerror = (e)=>{ log('Errorea chunk-ean:', e.error); chunkIndex++; speakNextChunk(); };
    window.speechSynthesis.speak(utterance);
  }

  function pickBestVoice(lang){
    return voices.find(v => v.lang === lang) || voices.find(v => v.lang.startsWith(lang.substring(0,2))) || voices[0];
  }

  function chunkText(text, maxLen){
    if(text.length <= maxLen) return [text];
    // split by sentence boundaries
    const sentences = text.match(/[^.!?]+[.!?]?/g) || [text];
    const chunks = [];
    let current = '';
    sentences.forEach(s => {
      const trimmed = s.trim();
      if((current + ' ' + trimmed).trim().length <= maxLen){
        current = (current ? current + ' ' : '') + trimmed;
      } else {
        if(current) chunks.push(current);
        if(trimmed.length > maxLen){
          // force split long sentence
          for(let i=0;i<trimmed.length;i+=maxLen){
            chunks.push(trimmed.slice(i,i+maxLen));
          }
          current='';
        } else {
          current = trimmed;
        }
      }
    });
    if(current) chunks.push(current);
    return chunks;
  }

  function highlight(el){
    clearHighlight();
    if(!el) return;
    el.dataset._origBg = el.style.backgroundColor || '';
    el.style.backgroundColor = '#fff3cd';
  }
  function clearHighlight(el){
    if(el){ el.style.backgroundColor = el.dataset._origBg || ''; return; }
    contentElements.forEach(e => { if(e.dataset._origBg !== undefined){ e.style.backgroundColor = e.dataset._origBg; } });
  }

  // Public API
  window.AudioReader = { init, start, stop, togglePlayPause };
})(window);

// Auto-init after DOM ready if .content exists
if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', ()=>{
    if(document.querySelector('.content')){ window.AudioReader.init({contentSelector: '.content'}); }
  });
} else {
  if(document.querySelector('.content')){ window.AudioReader.init({contentSelector: '.content'}); }
}

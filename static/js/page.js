(function () {
  'use strict';
  var D = window.PAGE_DATA;
  if (!D) return;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function pressOnly(container, btn) {
    container.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); });
  }

  function SyncGrid(grid, controls, onEnded, opts) {
    opts = opts || {};
    var vids = [], pending = null, token = 0;
    var playBtn = el('button', null, 'Pause'); playBtn.type = 'button';
    var range = el('input'); range.type = 'range'; range.min = 0; range.max = 1000; range.value = 0;
    var speed = el('select');
    [0.5, 1, 2].forEach(function (s) { var o = el('option', null, s + '×'); o.value = s; if (s === (opts.speed || 0.5)) o.selected = true; speed.appendChild(o); });
    var clock = el('span', 'pg-clock', 'Frame 1 / –');
    controls.append(playBtn, range, clock, speed);
    var playing = true, visible = false, scrubbing = false, manual = false, fps = 24, lastT = -1, lastMove = 0;

    function master() { return vids[0]; }
    function each(f) { vids.forEach(f); }
    function active() { return playing && visible; }
    function playAll() { each(function (v) { v.play().catch(function () {}); }); }
    function pauseAll() { each(function (v) { v.pause(); }); }
    function sync() { if (active()) playAll(); else pauseAll(); }

    playBtn.onclick = function () { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; sync(); };
    speed.onchange = function () { each(function (v) { v.playbackRate = +speed.value; }); };
    range.oninput = function () {
      var m = master(); if (!m || !m.duration) return;
      scrubbing = true; manual = true; pauseAll();
      var t = Math.min(range.value / 1000 * m.duration, m.duration - 2 / fps);
      each(function (v) { v.currentTime = t; });
    };
    range.onchange = function () { scrubbing = false; sync(); };

    var ticking = false;
    function startTick() { if (!ticking) { ticking = true; requestAnimationFrame(tick); } }
    new IntersectionObserver(function (es) { visible = es[0].isIntersecting; sync(); if (visible) startTick(); }, { threshold: 0.15 }).observe(grid);

    var near = false, deferred = null, api;
    new IntersectionObserver(function (es) {
      if (es[0].isIntersecting && !near) { near = true; if (deferred) { var d = deferred; deferred = null; api.load.apply(null, d); } }
    }, { rootMargin: '100% 0px' }).observe(grid);

    function onMasterEnded() {
      if (scrubbing) return;
      if (!manual && playing && onEnded) { onEnded(); return; }
      lastT = -1;
      each(function (v) { v.currentTime = 0; }); sync();
    }

    function tick() {
      var m = master();
      if (m && m.duration) {
        if (!scrubbing) range.value = Math.round(m.currentTime / m.duration * 1000);
        var total = Math.round(m.duration * fps);
        clock.textContent = 'Frame ' + Math.min(total, Math.floor(m.currentTime * fps + 1e-3) + 1) + ' / ' + total;

        var now = performance.now();
        if (m.currentTime !== lastT || m.paused || scrubbing) { lastT = m.currentTime; lastMove = now; }
        else if (now - lastMove > 400 && m.currentTime > m.duration - 0.25) { lastMove = now; onMasterEnded(); }
        if (!scrubbing && !m.paused) {
          for (var i = 1; i < vids.length; i++) {
            var v = vids[i];

            if (!v.seeking && Math.abs(v.currentTime - m.currentTime) > 0.15 && m.currentTime < v.duration - 0.05 && now - (v._nudge || 0) > 800) {
              v._nudge = now; v.currentTime = m.currentTime;
            }
          }
        }
      }
      if (visible || scrubbing) requestAnimationFrame(tick); else ticking = false;
    }

    api = {
      time: function () { return master() ? master().currentTime : 0; },
      load: function (panels, clipFps, startTime) {
        if (!near) { deferred = [panels, clipFps, startTime]; return; }
        var my = ++token;
        grid.setAttribute('aria-busy', 'true');
        if (pending) pending.forEach(function (v) { v.removeAttribute('src'); v.load(); });
        var frag = document.createDocumentFragment(), next = [];
        panels.forEach(function (p) {
          var isInput = /^(Input|Generated|GT$)/.test(p.label), framed = opts.frames !== false;
          var card = el('div', 'pg-panel' + (framed && p.ours ? ' is-ours' : '') + (framed && isInput ? ' is-input' : ''));
          if (p.video) {
            var v = el('video'); v.muted = true; v.playsInline = true; v.preload = 'auto';
            v.src = p.video + (p.video.indexOf('?') < 0 ? '?' : '&') + 'v=' + encodeURIComponent(D.version || '1'); v.playbackRate = +speed.value;
            card.appendChild(v); next.push(v);
          } else {
            card.classList.add('is-missing');
            card.appendChild(el('div', 'pg-missing', 'not available'));
          }
          card.appendChild(el('div', 'pg-panel-label', p.label));
          frag.appendChild(card);
        });
        pending = next;
        var ready = 0;
        function swap() {
          if (my !== token) return;
          pauseAll();
          grid.replaceChildren(frag);
          grid.setAttribute('aria-busy', 'false');
          vids = next; pending = null; fps = clipFps || 24; manual = false; scrubbing = false;
          if (startTime) each(function (v) { v.currentTime = Math.min(startTime, Math.max(0, v.duration - 1 / fps)); });
          if (vids[0]) vids[0].addEventListener('ended', onMasterEnded);
          sync();
        }
        if (!next.length) { swap(); return; }
        next.forEach(function (v) {
          v.addEventListener('loadeddata', function once() { v.removeEventListener('loadeddata', once); if (++ready === next.length) swap(); });
        });
      }
    };
    return api;
  }

  function chipSelector(chipsEl, items, labelOf, onPick) {
    items.forEach(function (it, k) {
      var b = el('button', null, labelOf(it)); b.type = 'button';
      b.onclick = function () { pressOnly(chipsEl, b); onPick(it, k); };
      chipsEl.appendChild(b);
    });
    var cur = 0;
    function pick(k) { cur = (k + items.length) % items.length; pressOnly(chipsEl, chipsEl.children[cur]); onPick(items[cur], cur); }
    Array.prototype.forEach.call(chipsEl.children, function (b, k) { b.onclick = function () { pick(k); }; });
    pick(0);
    return { next: function () { pick(cur + 1); } };
  }

  function tabbedSection(id, groups) {
    var root = document.getElementById(id);
    if (!root || !groups || !groups.length) return;
    groups = groups.filter(function (g) { return g.items.length; });
    if (groups.length < 2) root.querySelector('.pg-tabs').hidden = true;
    var tabs = root.querySelector('.pg-tabs'), label = root.querySelector('.pg-clip-label'), count = root.querySelector('.pg-clip-count');
    var g = 0, i = 0;
    var sync = SyncGrid(root.querySelector('.pg-sync-grid'), root.querySelector('.pg-controls'), function () {
      if (i + 1 < groups[g].items.length) show(g, i + 1); else show((g + 1) % groups.length, 0);
    });
    function show(gg, ii) {
      var items = groups[gg].items;
      g = gg; i = (ii + items.length) % items.length;
      pressOnly(tabs, tabs.children[g]);
      var it = items[i];
      label.textContent = it.label; count.textContent = (i + 1) + ' / ' + items.length;
      sync.load(it.panels.map(function (p) { return { video: p.video, label: p.label, ours: p.label === 'Ours' }; }), it.fps);
    }
    groups.forEach(function (grp, k) {
      var b = el('button', null, grp.title + ' (' + grp.items.length + ')'); b.type = 'button';
      b.onclick = function () { show(k, 0); };
      tabs.appendChild(b);
    });
    root.querySelector('.pg-clip-prev').onclick = function () { show(g, i - 1); };
    root.querySelector('.pg-clip-next').onclick = function () { show(g, i + 1); };
    show(0, 0);
  }

  (function () {
    var clips = window.LONGSEQ_DATA;
    if (!clips || !clips.length) return;
    var chips = document.getElementById('longseq-chips');
    var views = document.getElementById('longseq-views');
    var meta = document.getElementById('longseq-meta');
    var current = clips[0], view = 'input', selector;
    var sync = SyncGrid(document.getElementById('longseq-grid'), document.getElementById('longseq-controls'),
      function () { selector.next(); }, { speed: 1 });
    function show(time) {
      meta.textContent = current.frames + ' frames · ' + (current.frames / current.fps).toFixed(1) +
        ' seconds · ' + current.fps + ' FPS · sequence ' + current.name;
      sync.load(current.normalViews[view].map(function (p) {
        return { video: p.video, label: p.label, ours: p.label === 'Ours' };
      }), current.fps, time);
    }
    [['input', 'Input view'], ['side', 'Fixed side view']].forEach(function (item) {
      var b = el('button', null, item[1]); b.type = 'button';
      b.setAttribute('aria-pressed', item[0] === view ? 'true' : 'false');
      b.onclick = function () { var time = sync.time(); view = item[0]; pressOnly(views, b); show(time); };
      views.appendChild(b);
    });
    selector = chipSelector(chips, clips, function (c) { return c.label; }, function (c) { current = c; show(0); });
  })();

  (function () {
    if (!D.recon || !D.recon.length) return;
    var sel;
    var sync = SyncGrid(document.getElementById('recon-grid'), document.getElementById('recon-controls'), function () { sel.next(); });
    sel = chipSelector(document.getElementById('recon-chips'), D.recon, function (c) { return c.label; }, function (c) {
      sync.load(c.panels.map(function (p) { return { video: p.video, label: p.label, ours: p.label === 'Ours' }; }));
    });
  })();

  var wallVids = [], modalOpen = false;
  var ver = encodeURIComponent(D.version || '1');

  var nearObs = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      var v = e.target;
      if (e.isIntersecting && !v.getAttribute('src')) { v.src = v.dataset.src; v.load(); }
    });
  }, { rootMargin: '40% 0px' });

  function unload(v) { v.pause(); if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } }
  var visObs = new IntersectionObserver(function (es) {
    es.forEach(function (e) { e.target._vis = e.isIntersecting; wallSync(e.target); });
  }, { threshold: 0.4 });
  function wallSync(v) { if (v._vis && !modalOpen) v.play().catch(function () {}); else v.pause(); }

  var Modal = (function () {
    var modal = document.getElementById('real-modal');
    if (!modal) return null;
    var title = document.getElementById('real-modal-title'), grid = document.getElementById('real-grid');
    var sync = SyncGrid(grid, document.getElementById('real-controls'), null);
    var list = null, cur = -1;
    function show(k) {
      k = (k + list.length) % list.length; cur = k;
      var f = list[k], panels = f.panels;
      if (!modalOpen) { modalOpen = true; modal.hidden = false; document.body.classList.add('pg-modal-open'); wallVids.forEach(wallSync); }
      title.replaceChildren(el('b', null, f.set.title), document.createTextNode(' · ' + f.clip.label));
      grid.className = 'pg-sync-grid ' + (panels.length === 2 ? 'pg-cols-2' : panels.length <= 5 || panels.length >= 9 ? 'pg-cols-5' : panels.length === 6 ? 'pg-cols-3' : 'pg-cols-4');
      sync.load(panels.map(function (p) { return { video: p.video, label: p.label, ours: p.label === 'Ours' }; }), f.clip.fps);
    }
    function close() {
      if (!modalOpen) return;
      modalOpen = false; modal.hidden = true; document.body.classList.remove('pg-modal-open');
      grid.querySelectorAll('video').forEach(function (v) { v.pause(); });
      wallVids.forEach(wallSync);
    }
    document.getElementById('real-close').onclick = close;
    document.getElementById('real-prev').onclick = function () { show(cur - 1); };
    document.getElementById('real-next').onclick = function () { show(cur + 1); };
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    document.addEventListener('keydown', function (e) {
      if (!modalOpen) return;
      if (e.key === 'Escape') close();
      else if (e.key === 'ArrowLeft' && e.target.tagName !== 'INPUT') show(cur - 1);
      else if (e.key === 'ArrowRight' && e.target.tagName !== 'INPUT') show(cur + 1);
    });
    return { open: function (l, k) { list = l; show(k); } };
  })();

  function pairWall(wall, sets, cfg) {
    sets = sets.filter(function (x) { return x.items && x.items.length; });
    if (!wall || !Modal || !sets.length) return;
    var flat = [], SHOW = cfg.show;
    sets.forEach(function (set) {
      var head = el('div', 'pg-wall-head');
      head.append(el('h3', null, set.title), el('span', null, (set.note ? set.note + ' · ' : '') + set.items.length + ' clips'));
      var grid = el('div', 'pg-wall' + (cfg.cls ? ' ' + cfg.cls : ''));
      set.items.forEach(function (c, j) {
        var k = flat.length, panels = cfg.panels ? cfg.panels(c) : c.panels;
        flat.push({ set: set, clip: c, panels: panels });
        var card = el('button', 'pg-pair'); card.type = 'button';
        if (j >= SHOW) card.hidden = true;
        card.setAttribute('aria-label', set.title + ' ' + c.label + ': compare all methods');
        var media = el('div', 'pg-pair-media' + (cfg.left ? '' : ' is-single'));
        var v = el('video'); v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
        v.defaultPlaybackRate = 0.5; v.playbackRate = 0.5;
        v.addEventListener('loadedmetadata', function () { v.playbackRate = 0.5; });
        v.dataset.src = cfg.src(c) + '?v=' + ver;
        media.appendChild(v);
        if (cfg.left) media.appendChild(el('span', 'pg-pair-tag is-input', cfg.left));
        media.append(el('span', 'pg-pair-tag is-ours', 'Ours'), el('span', 'pg-pair-hint', 'Compare all methods'));
        var nb = panels.filter(function (p) { return !/^(Input|GT$|Ours$)/.test(p.label); }).length;
        var cap = el('div', 'pg-pair-cap');
        cap.append(el('span', null, c.label), el('span', 'pg-pair-more', nb + ' baselines ›'));
        card.append(media, cap);
        card.onclick = function () { Modal.open(flat, k); };
        grid.appendChild(card);
        wallVids.push(v); nearObs.observe(v); visObs.observe(v);
      });
      var wrap = el('div', 'pg-wall-wrap'); wrap.appendChild(grid);
      wall.append(head, wrap);

      var n = set.items.length, pages = Math.max(1, Math.ceil(n / SHOW)), pgi = 0;
      var count = el('span', 'pg-wall-count'); head.appendChild(count);
      var arrows = [['prev', '‹', -1, 'Previous'], ['next', '›', 1, 'Next']].map(function (x) {
        var btn = el('button', 'pg-wall-arrow is-' + x[0], x[1]); btn.type = 'button';
        btn.setAttribute('aria-label', x[3] + ' ' + set.title + ' clips');
        btn.onclick = function () { go(x[2]); };
        wrap.appendChild(btn); return btn;
      });
      function go(d) {
        pgi = Math.max(0, Math.min(pages - 1, pgi + d));
        var a0 = Math.max(0, Math.min(pgi * SHOW, n - SHOW));
        [].forEach.call(grid.children, function (c, j) {
          var hide = j < a0 || j >= a0 + SHOW;
          if (hide && !c.hidden) unload(c.querySelector('video'));
          c.hidden = hide;
        });
        count.textContent = (a0 + 1) + '–' + Math.min(n, a0 + SHOW) + ' of ' + n;
        arrows[0].disabled = pgi === 0; arrows[1].disabled = pgi === pages - 1;
      }
      go(0);
    });
  }

  pairWall(document.getElementById('real-wall'), [
    { title: 'DAVIS', note: 'Online videos of people and animals', items: (D.real || []).filter(function (x) { return x.name !== '602'; }) },
    { title: 'DyCheck', note: 'Handheld phone videos of everyday objects', items: window.PAGE_DYCHECK },
    { title: 'PhysTwin', note: 'Plush toys, cloth and rope moved by hand', items: window.PAGE_PHYSTWIN },
    { title: 'ARCTIC', note: 'Hands using articulated objects, calibrated camera', items: (D.arctic && D.arctic[0]) ? D.arctic[0].items : null }
  ], { show: 6, left: 'Input', src: function (c) { return 'static/videos/pairs/' + c.name + '.mp4'; } });

  pairWall(document.getElementById('compare-wall'), (D.compare || []).map(function (g) { return { title: g.title, items: g.items }; }), {
    show: 4, cls: 'pg-wall--4',
    src: function (c) { var o = c.panels.filter(function (p) { return p.label === 'Ours'; })[0]; return o.video; },
    panels: function (c) {
      var ps = c.panels.slice(), gt = { label: 'GT', video: 'static/videos/gen/' + c.name + '/gt.mp4' };
      ps.splice(ps[0] && /^Input/.test(ps[0].label) ? 1 : 0, 0, gt);
      return ps;
    }
  });

  (function () {
    var G = window.PAGE_GPT, tabs = document.getElementById('gpt-tabs'), chart = document.getElementById('gpt-chart');
    if (!G || !chart) return;

    function cell(v) {
      var c = el('div', 'pg-gpt-cell');
      if (!v) { c.appendChild(el('span', 'pg-gpt-na', 'not compared')); return c; }
      var n = v[1] + v[2] + v[3], bar = el('div', 'pg-gpt-bar');
      bar.title = v[1] + ' ours better, ' + v[2] + ' about the same, ' + v[3] + ' baseline better (of ' + n + ' segments)';
      [['is-win', v[1]], ['is-tie', v[2]], ['is-loss', v[3]]].forEach(function (x) {
        if (!x[1]) return;
        var seg = el('i', x[0], x[1] / n >= 0.12 ? String(x[1]) : ''); seg.style.flexGrow = x[1]; bar.appendChild(seg);
      });
      var sc = el('span', 'pg-gpt-score', v[0].toFixed(0) + '%');
      sc.title = 'Clip-averaged preference: ' + v[0].toFixed(1) + '%';
      c.append(bar, sc); return c;
    }
    function show(k) {
      pressOnly(tabs, tabs.children[k]);
      chart.innerHTML = '';
      var head = el('div', 'pg-gpt-row pg-gpt-head');
      head.append(el('div', 'pg-gpt-name', 'Ours vs.'), el('div', 'pg-gpt-cell', 'GPT judge · preference for ours'));
      chart.appendChild(head);
      G[k].rows.forEach(function (r) {
        var row = el('div', 'pg-gpt-row'), name = el('div', 'pg-gpt-name', r.label);
        name.appendChild(el('small', null, r.clips + ' clips'));
        row.append(name, cell(r.preference)); chart.appendChild(row);
      });

      var X = window.PAGE_GT3D; if (!X) return;
      var g = X[G[k].title], box = el('div', 'pg-gt3d');
      if (!g) return;
      box.appendChild(el('div', 'pg-gt3d-title', g.title + ' · ' + g.metric + ', lower is better'));
      box.appendChild(el('p', 'pg-gt3d-note', g.note + (g.source ? ' Source: ' + g.source + '.' : '')));
      var vals = [['Ours', g.ours, true]].concat(G[k].rows.filter(function (r) { return g.rows[r.label] != null; })
                   .map(function (r) { return [r.label, g.rows[r.label], false]; }));
      var best = Math.min.apply(null, vals.map(function (v) { return v[1]; })), hi = Math.max.apply(null, vals.map(function (v) { return v[1]; }));
      vals.forEach(function (v) {
        var row = el('div', 'pg-gt3d-row' + (v[2] ? ' is-ours' : '')), bar = el('div', 'pg-gt3d-bar'), fill = el('i');
        fill.style.width = (100 * v[1] / hi).toFixed(1) + '%'; bar.appendChild(fill);
        var num = el('span', 'pg-gt3d-num' + (v[1] === best ? ' is-best' : ''), v[1].toFixed(g.digits));
        row.append(el('div', 'pg-gt3d-name', v[0]), bar, num); box.appendChild(row);
      });
      chart.appendChild(box);
    }
    G.forEach(function (c, k) {
      var b = el('button', null, c.title + ' (' + c.sub + ')'); b.type = 'button'; b.onclick = function () { show(k); }; tabs.appendChild(b);
    });
    show(0);
  })();

  (function () {
    var grid = document.getElementById('failure-grid'), it = null;
    (D.real || []).forEach(function (x) { if (x.name === '602') it = x; });
    if (!grid || !it) return;
    var sync = SyncGrid(grid, document.getElementById('failure-controls'));
    sync.load(it.panels.map(function (p) { return { video: p.video, label: p.label, ours: p.label === 'Ours' }; }), it.fps);
  })();

  (function () {
    var wrap = document.querySelector('#insight .pg-insight-wrap');
    if (!wrap) return;
    var v = wrap.querySelector('video'), btn = wrap.querySelector('.pg-insight-play');
    btn.onclick = function () { v.controls = true; v.muted = false; v.play().catch(function () {}); };
    v.onclick = function () { if (!v.controls) btn.onclick(); };
    v.addEventListener('play', function () { wrap.classList.add('is-started'); });
    new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting && !v.paused) v.pause();
    }, { threshold: 0 }).observe(v);
  })();

  (function () {
    var links = Array.prototype.slice.call(document.querySelectorAll('#pg-nav a'));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var obs = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        var id = e.target.getAttribute('data-nav') || e.target.id;
        links.forEach(function (a) { a.classList.toggle('is-active', a === map[id]); });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) obs.observe(s); });
    Array.prototype.forEach.call(document.querySelectorAll('[data-nav]'), function (s) { obs.observe(s); });
  })();
})();

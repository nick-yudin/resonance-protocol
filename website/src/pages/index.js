import React, { useEffect } from 'react';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  useEffect(() => {
    // --- 1. TAILWIND SETUP (Safe Injection) ---
    if (!document.getElementById('tailwind-script')) {
      const script = document.createElement('script');
      script.id = 'tailwind-script';
      script.src = "https://cdn.tailwindcss.com";
      script.onload = () => {
        if (window.tailwind) {
          window.tailwind.config = {
            theme: {
              extend: {
                fontFamily: {
                  sans: ['Inter', 'sans-serif'],
                  display: ['Space Grotesk', 'sans-serif'],
                },
                colors: {
                  'void': '#050505',
                  'signal': '#ff4d00',
                  'noise': '#333333',
                  'silent': '#1a1a1a',
                },
                animation: {
                  'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                  'float': 'float 6s ease-in-out infinite',
                },
                keyframes: {
                  float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-20px)' },
                  }
                }
              }
            }
          };
        }
      };
      document.head.appendChild(script);
    }

    // --- 2. CANVAS VISUALIZER ---
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];

    const PARTICLE_COUNT = 50;
    const CONNECTION_DIST = 150;
    const SIGNAL_COLOR = '255, 77, 0'; 
    const NOISE_COLOR = '255, 255, 255'; 

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    class Particle {
      constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.meaning = 0; 
        this.timer = 0;
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;

        if (Math.random() < 0.005 && this.meaning === 0) {
          this.meaning = 1;
          this.timer = 100; 
        }

        if (this.meaning === 1) {
          this.timer--;
          if (this.timer <= 0) this.meaning = 0;
        }
      }

      draw() {
        ctx.beginPath();
        const alpha = this.meaning === 1 ? 0.8 : 0.1;
        const color = this.meaning === 1 ? SIGNAL_COLOR : NOISE_COLOR;
        ctx.fillStyle = `rgba(${color}, ${alpha})`;
        ctx.arc(this.x, this.y, this.meaning === 1 ? 3 : 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function init() {
      resize();
      particles = [];
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
      }
    }

    function animate() {
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < particles.length; i++) {
        let p1 = particles[i];
        p1.update();
        p1.draw();

        for (let j = i + 1; j < particles.length; j++) {
          let p2 = particles[j];
          let dx = p1.x - p2.x;
          let dy = p1.y - p2.y;
          let dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < CONNECTION_DIST) {
            if (p1.meaning === 1 && p2.meaning === 1) {
              ctx.beginPath();
              ctx.strokeStyle = `rgba(${SIGNAL_COLOR}, ${1 - dist / CONNECTION_DIST})`;
              ctx.lineWidth = 2;
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
              ctx.stroke();
            } else {
              ctx.beginPath();
              ctx.strokeStyle = `rgba(${NOISE_COLOR}, 0.05)`;
              ctx.lineWidth = 0.5;
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
              ctx.stroke();
            }
          }
        }
      }
      requestAnimationFrame(animate);
    }

    window.addEventListener('resize', resize);
    init();
    const animId = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <div className="antialiased selection:bg-signal selection:text-white bg-[#050505] min-h-screen text-[#e5e5e5]">
      <Head>
        <title>RESONANCE | The Quiet Protocol</title>
        <meta name="description" content="A protocol for semantic event computing." />
        
        {/* Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet" />
        
        {/* Custom Styles */}
        <style>{`
          body { background-color: #050505; color: #e5e5e5; overflow-x: hidden; margin: 0; }
          .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); }
          .glow-text { text-shadow: 0 0 20px rgba(255, 77, 0, 0.5); }
          .gradient-text { background: linear-gradient(to right, #fff, #666); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
          
          /* Custom Scrollbar */
          ::-webkit-scrollbar { width: 8px; }
          ::-webkit-scrollbar-track { background: #050505; }
          ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
          ::-webkit-scrollbar-thumb:hover { background: #ff4d00; }

          canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }
        `}</style>
      </Head>

      {/* Navigation */}
      <nav className="fixed w-full z-50 top-0 py-4 px-6 bg-void/80 backdrop-blur-md border-b border-white/5">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="text-2xl font-display font-bold tracking-tighter text-white flex items-center gap-2">
            <div className="w-3 h-3 bg-signal rounded-full animate-pulse"></div>
            RESONANCE
          </div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-gray-400">
            <a href="#shift" className="hover:text-signal transition-colors">The Shift</a>
            <Link to="/docs/manifesto" className="hover:text-signal transition-colors">Manifesto</Link>
            <a href="#future" className="hover:text-signal transition-colors">Future History</a>
            <Link to="/docs/unified-spec" className="text-white border border-white/20 px-4 py-1.5 rounded-full hover:bg-white hover:text-black transition-all">
              Read Protocol
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section with Visualizer */}
      <section className="relative h-screen flex flex-col justify-center items-center px-6 overflow-hidden">
        
        {/* Canvas Background */}
        <canvas id="hero-canvas"></canvas>

        <div className="relative z-10 max-w-5xl text-center space-y-8 pt-20">
          <div className="inline-block px-3 py-1 border border-signal/30 rounded text-xs font-mono text-signal tracking-widest uppercase mb-4 animate-fade-in">
            Protocol Level 1 // Unified Spec
          </div>
          <h1 className="text-5xl md:text-8xl font-display font-bold leading-tight tracking-tight text-white">
            The Clock Stops.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-600">The Resonance Begins.</span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto font-light leading-relaxed">
            We are replacing the tyranny of time with the physics of meaning. A paradigm shift to <span className="text-white">Ambient AGI</span>—ubiquitous as air, personal as a memory.
          </p>
          
          <div className="flex flex-col md:flex-row justify-center gap-4 mt-12">
            <Link to="/docs/unified-spec" className="group px-8 py-4 bg-white text-black font-bold rounded hover:bg-gray-200 transition-all flex items-center justify-center gap-2">
              Read Specification
              <svg className="group-hover:translate-y-1 transition-transform" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17"/></svg>
            </Link>
            <a href="#shift" className="px-8 py-4 border border-white/20 text-white rounded hover:border-signal hover:text-signal transition-all">
              Explore the Shift
            </a>
          </div>
        </div>
        
        {/* Scroll Indicator */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-bounce text-gray-600">
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
        </div>
      </section>

      {/* Section: The Shift */}
      <section id="shift" className="py-32 px-6 bg-silent relative">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-20 items-center">
            <div>
              <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6">The Shift</h2>
              <h3 className="text-4xl md:text-5xl font-display font-bold text-white mb-8">From Time-Based to Meaning-Based Computing</h3>
              <div className="space-y-8">
                <div className="border-l-2 border-gray-700 pl-6 py-2">
                  <h4 className="text-gray-500 font-mono text-sm mb-1">OLD PARADIGM</h4>
                  <p className="text-2xl text-gray-400">"Compute every nanosecond, whether needed or not."</p>
                  <p className="text-sm text-gray-600 mt-2">Result: Energy waste, heat, latency.</p>
                </div>
                <div className="border-l-2 border-signal pl-6 py-2">
                  <h4 className="text-signal font-mono text-sm mb-1">NEW PARADIGM</h4>
                  <p className="text-2xl text-white">"Compute only when meaning changes."</p>
                  <p className="text-sm text-gray-400 mt-2">Result: Native efficiency, sparsity, instant reflex.</p>
                </div>
              </div>
            </div>
            <div className="relative">
              <div className="absolute -inset-4 bg-signal/10 blur-3xl rounded-full"></div>
              <div className="glass p-8 rounded-2xl relative overflow-hidden">
                <div className="flex flex-col gap-4 font-mono text-xs">
                  <div className="flex justify-between border-b border-white/10 pb-2 text-gray-500">
                    <span>CLOCK CYCLE 1</span> <span>COMPUTING 0...</span>
                  </div>
                  <div className="flex justify-between border-b border-white/10 pb-2 text-gray-500">
                    <span>CLOCK CYCLE 2</span> <span>COMPUTING 0...</span>
                  </div>
                  <div className="flex justify-between border-b border-white/10 pb-2 text-gray-500">
                    <span>CLOCK CYCLE 3</span> <span>COMPUTING 0...</span>
                  </div>
                  <div className="flex justify-between border-b border-signal pb-2 text-signal font-bold">
                    <span>EVENT DETECTED</span> <span>MEANING UPDATE!</span>
                  </div>
                  <div className="flex justify-between border-b border-white/10 pb-2 text-gray-500 opacity-50">
                    <span>SILENCE...</span> <span>SLEEPING</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section: Manifesto (4 Axioms) */}
      <section id="manifesto" className="py-32 px-6 border-t border-white/5 bg-[#050505]">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-center text-4xl font-display font-bold text-white mb-20">The Four Axioms</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="glass p-8 rounded-xl hover:border-signal/50 transition-colors group">
              <div className="text-signal font-mono text-4xl mb-4 opacity-50 group-hover:opacity-100 transition-opacity">01</div>
              <h3 className="text-xl font-bold text-white mb-4">Intelligence is Air</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Intelligence should be a property of the environment, not a privilege of the few. It must be ubiquitous, invisible, and accessible like oxygen.
              </p>
            </div>
            <div className="glass p-8 rounded-xl hover:border-signal/50 transition-colors group">
              <div className="text-signal font-mono text-4xl mb-4 opacity-50 group-hover:opacity-100 transition-opacity">02</div>
              <h3 className="text-xl font-bold text-white mb-4">Energy {'>'} Compute</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Efficiency is the ultimate benchmark. If a system consumes gigawatts to think, it is not intelligent; it is just a furnace.
              </p>
            </div>
            <div className="glass p-8 rounded-xl hover:border-signal/50 transition-colors group">
              <div className="text-signal font-mono text-4xl mb-4 opacity-50 group-hover:opacity-100 transition-opacity">03</div>
              <h3 className="text-xl font-bold text-white mb-4">Distribution {'>'} Control</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Robustness comes from the mesh, not the monolith. A centralized brain is a single point of failure and control.
              </p>
            </div>
            <div className="glass p-8 rounded-xl hover:border-signal/50 transition-colors group">
              <div className="text-signal font-mono text-4xl mb-4 opacity-50 group-hover:opacity-100 transition-opacity">04</div>
              <h3 className="text-xl font-bold text-white mb-4">Privacy is Physics</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Data sovereignty must be guaranteed by hardware constraints (air-gapped semantics), not by legal promises or user agreements.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section: Future History (Fictions) */}
      <section id="future" className="py-32 px-6 bg-gray-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-end justify-between mb-16">
            <div>
              <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-2">Future History</h2>
              <h3 className="text-3xl font-display font-bold text-white">Artifacts from 2030</h3>
            </div>
            <div className="hidden md:block text-right text-xs font-mono text-gray-500">
              LOCATION: EARTH<br />STATUS: RESONANT
            </div>
          </div>

          <div className="space-y-24">
            
            {/* Story 1: The Day the Cloud Failed */}
            <div className="grid md:grid-cols-2 gap-12 items-center group">
              <div className="order-2 md:order-1">
                <div className="text-xs font-mono text-signal mb-2">CASE: THE DAY THE CLOUD FAILED</div>
                <h4 className="text-2xl font-bold text-white mb-4">Local Intelligence, Global Outage</h4>
                <p className="text-gray-400 leading-relaxed mb-6">
                  "For six hours, every major cloud provider went dark. No LLM APIs, no centralized vision services, no auth tokens. Dashboards screamed, markets panicked — but the systems that mattered didn't. Ports kept routing containers. Hospital triage lines kept moving. Traffic lights adapted to real flows. None of these systems were calling home. Their models lived at the edge; their decisions emerged from local semantic events, not remote inference calls. For the first time, the cloud disappeared — and the world did not."
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Cloud Dependency: 0%</span>
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Mode: Local-Only</span>
                </div>
              </div>
              <div className="order-1 md:order-2 h-64 bg-noise rounded-lg border border-white/5 relative overflow-hidden group-hover:border-signal/30 transition-colors">
                <img src="/case1_s.jpg" alt="Local Intelligence" className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700" />
                <div className="absolute inset-0 bg-black/40"></div>
                <div className="absolute inset-0 flex items-center justify-center text-white/80 font-mono text-sm z-10">[SEMANTIC MESH: SERVICES STABLE]</div>
                <div className="absolute bottom-0 left-0 w-full h-1 bg-signal/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 z-10"></div>
              </div>
            </div>

            {/* Story 2: The City With No Cameras */}
            <div className="grid md:grid-cols-2 gap-12 items-center group">
              <div className="h-64 bg-noise rounded-lg border border-white/5 relative overflow-hidden group-hover:border-signal/30 transition-colors">
                <img src="/case2_s.jpg" alt="City with no cameras" className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700" />
                <div className="absolute inset-0 bg-black/40"></div>
                <div className="absolute inset-0 flex items-center justify-center text-white/80 font-mono text-sm z-10">[SEMANTIC FIELD: NO DATA STORED]</div>
                <div className="absolute bottom-0 left-0 w-full h-1 bg-signal/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 z-10"></div>
              </div>
              <div>
                <div className="text-xs font-mono text-signal mb-2">CASE: THE CITY WITH NO CAMERAS</div>
                <h4 className="text-2xl font-bold text-white mb-4">Public Safety Without Surveillance</h4>
                <p className="text-gray-400 leading-relaxed mb-6">
                  "Ten years ago, the metropolis was blanketed with CCTV — a nervous system wired straight into storage. Today there are no 'cameras' in the old sense. Intersections still see. Stations still notice. Streets still react. But nothing is recorded, nothing uploaded, nothing stored. Hardware enforces it: only semantic events exist, and they decay on-device. A fight starts? Local nodes trigger light, sound and nearby responders. A lost child? The mesh quietly guides parents — without ever building a face database. Crime went down. Data hoarding fell to zero. The city became attentive, not voyeuristic."
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Video Stored: 0 Frames</span>
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Surveillance: Replaced</span>
                </div>
              </div>
            </div>

            {/* Story 3: The Body That Diagnosed Itself */}
            <div className="grid md:grid-cols-2 gap-12 items-center group">
              <div className="order-2 md:order-1">
                <div className="text-xs font-mono text-signal mb-2">CASE: THE BODY THAT DIAGNOSED ITSELF</div>
                <h4 className="text-2xl font-bold text-white mb-4">30-Second Semantic Medicine</h4>
                <p className="text-gray-400 leading-relaxed mb-6">
                  "People used to wear 'health trackers' that streamed everything to the cloud. It was convenient, but it was surveillance. The new generation tracks nothing. No continuous recording, no streaming. A tiny semantic model runs on-skin. Millions of micro-signals stay local. When patterns drift into meaningful territory — a perfusion mismatch, a metabolic stress vector — the wearable emits a single semantic event. Not data, meaning: 'Act Now'. You walk into a clinic. No upload. Your wearable and the room’s mesh converge on a diagnosis in thirty seconds. The device never gives away your secrets. It never even had them."
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Wearables: Local-Only</span>
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Diagnosis: 30 Seconds</span>
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Privacy: Absolute</span>
                </div>
              </div>
              <div className="order-1 md:order-2 h-64 bg-noise rounded-lg border border-white/5 relative overflow-hidden group-hover:border-signal/30 transition-colors">
                <img src="/case3_s.jpg" alt="Medical diagnosis" className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700" />
                <div className="absolute inset-0 bg-black/40"></div>
                <div className="absolute inset-0 flex items-center justify-center text-white/80 font-mono text-sm z-10">[SEMANTIC EVENTS: TRUE CRITICAL]</div>
                <div className="absolute bottom-0 left-0 w-full h-1 bg-signal/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 z-10"></div>
              </div>
            </div>

            {/* Story 4: The Last Language That Survived */}
            <div className="grid md:grid-cols-2 gap-12 items-center group">
              <div className="h-64 bg-noise rounded-lg border border-white/5 relative overflow-hidden group-hover:border-signal/30 transition-colors">
                <img src="/case4_s.jpg" alt="Language preservation" className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700" />
                <div className="absolute inset-0 bg-black/40"></div>
                <div className="absolute inset-0 flex items-center justify-center text-white/80 font-mono text-sm z-10">[SEMANTIC CORE: LANGUAGE LIVES LOCALLY]</div>
                <div className="absolute bottom-0 left-0 w-full h-1 bg-signal/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 z-10"></div>
              </div>
              <div>
                <div className="text-xs font-mono text-signal mb-2">CASE: THE LAST LANGUAGE THAT SURVIVED</div>
                <h4 className="text-2xl font-bold text-white mb-4">Endangered Tongues, Embedded in Silicon</h4>
                <p className="text-gray-400 leading-relaxed mb-6">
                  "There used to be a joke: 'If it's not in the cloud, it doesn't exist.' That joke killed a thousand languages. No venture-backed model would ever be trained on a village that couldn't pay. With Resonance, the direction flipped. A handful of solar-powered nodes, deep in a mountain valley, listened and learned locally for years — never uploading a byte. Children spoke to grandparents; the mesh distilled patterns into a tiny semantic core that lives only there. Now when a child speaks in the dominant language, the answer arrives in the ancestral one — instantly, offline. No cloud owns it. No corporation can deprecate the API. As long as the village keeps its devices alive, the language remains alive — not as an archive, but as a living interface."
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Connectivity: Offline</span>
                  <span className="px-2 py-1 bg-white/10 rounded text-xs text-gray-300">Ownership: Local</span>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 border-t border-white/10 px-6 bg-void">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="text-left">
            <div className="text-xl font-bold text-white mb-2 tracking-tighter">RESONANCE</div>
            <p className="text-gray-500 text-sm max-w-xs">
              An open protocol for the post-clock era of computing. <br />
              Not a product. Infrastructure.
            </p>
          </div>
          
          <div className="flex gap-6 font-mono text-sm">
            <a href="https://twitter.com/rAI_stack" target="_blank" className="text-gray-500 hover:text-white transition-colors">Twitter / X</a>
            <a href="https://github.com/nick-yudin/resonance-protocol" className="text-gray-500 hover:text-white transition-colors">GitHub</a>
            <a href="mailto:1@resonanceprotocol.org" className="text-gray-500 hover:text-white transition-colors">Contact</a>
          </div>
        </div>
        <div className="max-w-6xl mx-auto mt-12 pt-8 border-t border-white/5 text-center text-xs text-gray-800 font-mono">
          INITIATED 2025 // rAI RESEARCH COLLECTIVE // SILENCE IS GOLDEN
        </div>
      </footer>
    </div>
  );
}

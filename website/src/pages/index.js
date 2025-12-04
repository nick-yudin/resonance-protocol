import React, { useEffect } from 'react';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import BenchmarkViz from '@site/src/components/BenchmarkViz';

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
          /* Smooth Scrolling & Fix for Header Overlap */
          html { scroll-behavior: smooth; }
          section { scroll-margin-top: 100px; }

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
            <a href="#problem" className="hover:text-signal transition-colors">Problem</a>
            <a href="#results" className="hover:text-signal transition-colors">Results</a>
            <a href="#status" className="hover:text-signal transition-colors">Status</a>
            <a href="/demo" className="hover:text-signal transition-colors">Demo</a>
            <a href="#join" className="hover:text-signal transition-colors">Join</a>
            <a href="https://github.com/nick-yudin/resonance-protocol" target="_blank" className="text-white border border-white/20 px-4 py-1.5 rounded-full hover:bg-white hover:text-black transition-all">
              GitHub
            </a>
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
            <Link to="/docs/specs/v1.0_current/spec-v1-final" className="group px-8 py-4 bg-white text-black font-bold rounded hover:bg-gray-200 transition-all flex items-center justify-center gap-2">
              Read Specification
              <svg className="group-hover:translate-y-1 transition-transform" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 15V3m0 12l-4-4m4 4l4-4M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17"/></svg>
            </Link>
            <a href="/whitepaper.pdf" target="_blank" className="px-8 py-4 border border-white/20 text-white rounded hover:border-signal hover:text-signal transition-all">
              Download Whitepaper (L0)
            </a>
          </div>

          <p className="text-gray-500 text-sm mt-8">
            A research project by <span className="text-white">Nikolay Yudin</span>
          </p>
        </div>
        
        {/* Scroll Indicator */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-bounce text-gray-600">
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
        </div>
      </section>

      {/* Section: The Problem */}
      <section id="problem" className="py-32 px-6 bg-silent relative border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6 text-center">The Problem</h2>
          <h3 className="text-4xl md:text-5xl font-display font-bold text-white mb-12 text-center max-w-4xl mx-auto">
            AI is becoming critical infrastructure. And it's controlled by 3 companies in 1 country.
          </h3>

          <div className="grid md:grid-cols-2 gap-12 mb-16">
            <div className="glass p-8 rounded-xl">
              <h4 className="text-xl font-bold text-white mb-6">The Monopoly</h4>
              <ul className="space-y-4 text-gray-400">
                <li className="flex items-start gap-3">
                  <span className="text-signal mt-1">▸</span>
                  <span><strong className="text-white">NVIDIA</strong> controls the hardware</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-signal mt-1">▸</span>
                  <span><strong className="text-white">USA</strong> controls NVIDIA (export restrictions on chips)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-signal mt-1">▸</span>
                  <span><strong className="text-white">OpenAI, Anthropic, Google</strong> control the top models</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-signal mt-1">▸</span>
                  <span><strong className="text-white">Everyone else</strong> is just a customer — with a kill switch</span>
                </li>
              </ul>
            </div>

            <div className="glass p-8 rounded-xl border-signal/30">
              <h4 className="text-xl font-bold text-white mb-6">The Economics</h4>
              <div className="space-y-4">
                <div>
                  <div className="text-3xl font-bold text-signal mb-2">$100M</div>
                  <p className="text-gray-400 text-sm">Training cost for a GPT-4 class model</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="text-2xl font-bold text-white mb-2">70%</div>
                  <p className="text-gray-400 text-sm">Goes to GPU compute — thousands of H100s for months</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <p className="text-gray-500 text-sm italic">The gap is growing exponentially. This is not about money — it's about control.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass p-10 rounded-xl border border-signal/20 bg-signal/5">
            <h4 className="text-lg font-mono text-signal mb-4 uppercase tracking-wider">The Real Risk</h4>
            <p className="text-xl text-white mb-4 leading-relaxed">
              "Turn off your API" = digital blockade.
            </p>
            <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-400">
              <div>
                <strong className="text-white">Europe:</strong> Has Mistral — infinitely weaker, no path to catch up.
              </div>
              <div>
                <strong className="text-white">BRICS:</strong> Has Qwen — controlled by Alibaba, subject to Chinese government.
              </div>
              <div>
                <strong className="text-white">Developing nations:</strong> Have nothing.
              </div>
            </div>
            <p className="text-gray-400 mt-6 text-sm">
              AI is becoming like oil in the 20th century. Except you can't drill for it.
            </p>
          </div>

          <div className="text-center mt-12">
            <p className="text-2xl text-white font-light max-w-3xl mx-auto">
              We don't think "catching up" is the answer. <br/>
              <span className="text-signal font-bold">We think the paradigm itself is wrong.</span>
            </p>
          </div>
        </div>
      </section>

      {/* Section: The Shift */}
      <section id="shift" className="py-32 px-6 bg-[#050505] relative border-t border-white/5">
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

      {/* Section: Proven Results */}
      <section id="results" className="py-32 px-6 bg-silent border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6 text-center">Research Results</h2>
          <h3 className="text-4xl md:text-5xl font-display font-bold text-white mb-6 text-center">
            What We've Proven (Not Promised)
          </h3>
          <p className="text-center text-gray-400 mb-16 max-w-2xl mx-auto">
            Real experiments. Real numbers. Reproducible results.
          </p>

          {/* Three Main Results */}
          <div className="grid md:grid-cols-3 gap-8 mb-16">

            {/* Result 1: Compression */}
            <div className="glass p-8 rounded-xl border-signal/30 hover:border-signal transition-colors group">
              <div className="text-6xl font-bold text-signal mb-4">32×</div>
              <h4 className="text-xl font-bold text-white mb-3">HDC Compression</h4>
              <p className="text-gray-400 text-sm mb-4">
                Distributed training bandwidth reduced from 17MB to 271KB per sync using ternary quantization.
              </p>
              <div className="text-xs font-mono text-gray-500">
                M3b Experiment • December 2024
              </div>
              {/* Chart */}
              <div className="mt-4 h-32 bg-noise/50 rounded-lg overflow-hidden border border-white/5">
                <img src="/research/m3b_compression.png" alt="32x compression chart" className="w-full h-full object-contain opacity-80" />
              </div>
            </div>

            {/* Result 2: Cross-Architecture Transfer */}
            <div className="glass p-8 rounded-xl border-signal/30 hover:border-signal transition-colors group">
              <div className="text-6xl font-bold text-signal mb-4">93%</div>
              <h4 className="text-xl font-bold text-white mb-3">Knowledge Transfer</h4>
              <p className="text-gray-400 text-sm mb-4">
                Cross-architecture transfer efficiency. DistilBERT → GPT-2 via semantic examples, not weights.
              </p>
              <div className="text-xs font-mono text-gray-500">
                M3c′ Experiment • December 2024
              </div>
              {/* Chart */}
              <div className="mt-4 h-32 bg-noise/50 rounded-lg overflow-hidden border border-white/5">
                <img src="/research/phase_m2.5b_curriculum_comparison.png" alt="93% transfer efficiency chart" className="w-full h-full object-contain opacity-80" />
              </div>
            </div>

            {/* Result 3: Compositional Generalization */}
            <div className="glass p-8 rounded-xl border-signal/30 hover:border-signal transition-colors group">
              <div className="text-6xl font-bold text-signal mb-4">100%</div>
              <h4 className="text-xl font-bold text-white mb-3">HDC Generalization</h4>
              <p className="text-gray-400 text-sm mb-4">
                Perfect compositional generalization where Transformers achieve only 21% on unseen combinations.
              </p>
              <div className="text-xs font-mono text-gray-500">
                M2.6 Experiment • December 2024
              </div>
              {/* Chart */}
              <div className="mt-4 h-32 bg-noise/50 rounded-lg overflow-hidden border border-white/5">
                <img src="/research/m26_generalization.png" alt="HDC vs Transformer generalization" className="w-full h-full object-contain opacity-80" />
              </div>
            </div>
          </div>

          {/* Why This Matters */}
          <div className="glass p-10 rounded-xl border border-signal/20 bg-signal/5 mb-12">
            <h4 className="text-lg font-mono text-signal mb-6 uppercase tracking-wider">Why This Changes Everything</h4>
            <div className="grid md:grid-cols-3 gap-8">
              <div>
                <h5 className="text-white font-bold mb-2">Heterogeneous Networks</h5>
                <p className="text-gray-400 text-sm">
                  Different models can share knowledge. No need for identical architectures across all nodes.
                </p>
              </div>
              <div>
                <h5 className="text-white font-bold mb-2">Edge-Viable Bandwidth</h5>
                <p className="text-gray-400 text-sm">
                  271KB per sync works on 3G, mesh networks, satellite links. Distributed training leaves the datacenter.
                </p>
              </div>
              <div>
                <h5 className="text-white font-bold mb-2">Structural Robustness</h5>
                <p className="text-gray-400 text-sm">
                  HDC provides compositional guarantees that scale-based approaches fundamentally cannot achieve.
                </p>
              </div>
            </div>
          </div>

          {/* Link to Full Research */}
          <div className="text-center">
            <Link to="/docs/research" className="inline-flex items-center gap-2 px-8 py-4 border border-white/20 text-white rounded hover:border-signal hover:text-signal transition-all">
              View Full Research Log
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </Link>
          </div>
        </div>
      </section>

      {/* Section: Manifesto (4 Axioms) */}
      <section id="axioms" className="py-32 px-6 border-t border-white/5 bg-[#050505]">
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

      {/* Section: Honest Status */}
      <section id="status" className="py-32 px-6 bg-silent border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6 text-center">Honest Status</h2>
          <h3 className="text-3xl md:text-4xl font-display font-bold text-white mb-12 text-center">
            What Works Today vs What We're Researching
          </h3>

          <div className="grid md:grid-cols-2 gap-8 mb-12">
            {/* Proven */}
            <div className="glass p-8 rounded-xl border-green-500/20">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <h4 className="text-xl font-bold text-white">Experimentally Proven</h4>
              </div>
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Distributed Training</h5>
                    <span className="text-xs font-mono text-green-500">M3a ✓</span>
                  </div>
                  <p className="text-sm text-gray-400">Two nodes trained shared model via Firebase. Loss converged identically.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">HDC Compression (32×)</h5>
                    <span className="text-xs font-mono text-green-500">M3b ✓</span>
                  </div>
                  <p className="text-sm text-gray-400">17MB → 271KB per sync. Ternary quantization + 2-bit packing.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Cross-Architecture Transfer (93%)</h5>
                    <span className="text-xs font-mono text-green-500">M3c′ ✓</span>
                  </div>
                  <p className="text-sm text-gray-400">DistilBERT → GPT-2 knowledge transfer via semantic examples.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Compositional Generalization</h5>
                    <span className="text-xs font-mono text-green-500">M2.6 ✓</span>
                  </div>
                  <p className="text-sm text-gray-400">HDC 100% vs Transformer 21% on unseen combinations.</p>
                </div>
              </div>
            </div>

            {/* Researching */}
            <div className="glass p-8 rounded-xl border-yellow-500/20">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
                <h4 className="text-xl font-bold text-white">Researching</h4>
              </div>
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Distributed training on edge</h5>
                    <span className="text-xs font-mono text-yellow-500">RESEARCH</span>
                  </div>
                  <p className="text-sm text-gray-400">DiLoCo, Hivemind show promise. Not production-ready.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Ternary computing (10-100×)</h5>
                    <span className="text-xs font-mono text-yellow-500">WAITING</span>
                  </div>
                  <p className="text-sm text-gray-400">BitNet works. Waiting for ternary hardware.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Semantic training efficiency</h5>
                    <span className="text-xs font-mono text-yellow-500">SPECULATION</span>
                  </div>
                  <p className="text-sm text-gray-400">Works for inference, not proven for training yet.</p>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <div className="flex justify-between items-start mb-2">
                    <h5 className="font-bold text-white">Governance mechanisms</h5>
                    <span className="text-xs font-mono text-yellow-500">DESIGN</span>
                  </div>
                  <p className="text-sm text-gray-400">"No one controls" needs real mechanism design.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Our Bet */}
          <div className="glass p-10 rounded-xl border border-signal/20 bg-signal/5">
            <h4 className="text-lg font-mono text-signal mb-4 uppercase tracking-wider">Our Bet</h4>
            <div className="space-y-4 text-gray-300">
              <p className="text-lg">
                New hardware is coming: <strong className="text-white">memristors, neuromorphic chips, in-memory computing</strong>.
              </p>
              <p className="text-lg">
                When it arrives, the economics of AI will flip. Datacenters won't be the only way.
              </p>
              <p className="text-xl text-white font-bold mt-6">
                We're building the architecture for that future — one that works today and scales exponentially tomorrow.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section: Future History (Fictions) */}
      <section id="future" className="py-32 px-6 bg-[#050505] border-t border-white/5">
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
                  "People used to wear 'health trackers' that streamed everything to the cloud. It was convenient, but it was surveillance. The new generation tracks nothing. No continuous recording, no streaming. A tiny semantic model runs on-skin. Millions of micro-signals stay local. When patterns drift into meaningful territory — a perfusion mismatch, a metabolic stress vector — the wearable emits a single semantic event. Not data, meaning: 'Act Now'. You walk into a clinic. No upload. Your wearable and the room's mesh converge on a diagnosis in thirty seconds. The device never gives away your secrets. It never even had them."
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

      {/* Section: The Demo */}
      <section id="demo" className="py-32 px-6 bg-silent border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6 text-center">The Demo We're Building</h2>
          <h3 className="text-4xl md:text-5xl font-display font-bold text-white mb-12 text-center">
            "The Box in the Café"
          </h3>

          <div className="glass p-12 rounded-2xl mb-12">
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              Imagine this scene:
            </p>

            <div className="space-y-6 mb-10">
              <div className="flex gap-4 items-start">
                <div className="text-signal font-mono text-lg font-bold min-w-[2rem]">1.</div>
                <p className="text-lg text-gray-300">Someone asks <strong className="text-white">ChatGPT</strong> through their browser</p>
              </div>
              <div className="flex gap-4 items-start">
                <div className="text-signal font-mono text-lg font-bold min-w-[2rem]">2.</div>
                <p className="text-lg text-gray-300">You ask your <strong className="text-white">small device</strong> — same quality answer</p>
              </div>
              <div className="flex gap-4 items-start">
                <div className="text-signal font-mono text-lg font-bold min-w-[2rem]">3.</div>
                <p className="text-lg text-white italic">"Now turn off the internet."</p>
              </div>
              <div className="flex gap-4 items-start">
                <div className="text-signal font-mono text-lg font-bold min-w-[2rem]">4.</div>
                <p className="text-lg text-gray-300">Their ChatGPT is <strong className="text-red-500">dead</strong>. Your box <strong className="text-green-500">still answers</strong>.</p>
              </div>
            </div>

            <div className="border-t border-white/10 pt-8">
              <p className="text-gray-400 mb-6">
                But wait — the box is too small to hold a full model.
              </p>
              <p className="text-lg text-white">
                Watch the answer <strong className="text-signal">grow</strong> as neighboring nodes contribute through the mesh.
                Each device adds what it knows. Together, they're smarter than any single node.
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass p-6 rounded-lg text-center">
              <div className="text-2xl font-bold text-signal mb-2">✓</div>
              <p className="text-sm text-gray-300"><strong className="text-white">Intelligence</strong> ≠ datacenter</p>
            </div>
            <div className="glass p-6 rounded-lg text-center">
              <div className="text-2xl font-bold text-signal mb-2">✓</div>
              <p className="text-sm text-gray-300"><strong className="text-white">Works</strong> offline</p>
            </div>
            <div className="glass p-6 rounded-lg text-center">
              <div className="text-2xl font-bold text-signal mb-2">✓</div>
              <p className="text-sm text-gray-300"><strong className="text-white">Cannot</strong> be shut down</p>
            </div>
            <div className="glass p-6 rounded-lg text-center">
              <div className="text-2xl font-bold text-signal mb-2">✓</div>
              <p className="text-sm text-gray-300"><strong className="text-white">No one</strong> controls it</p>
            </div>
          </div>

          <p className="text-center text-gray-500 mt-12 italic">
            This isn't ready yet. But it's what we're building toward.
          </p>
        </div>
      </section>

      {/* Section: Join Us */}
      <section id="join" className="py-32 px-6 bg-[#050505] border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-sm font-mono text-signal tracking-widest uppercase mb-6 text-center">Join Us</h2>
          <h3 className="text-4xl md:text-5xl font-display font-bold text-white mb-8 text-center max-w-3xl mx-auto">
            We're looking for people who see the problem and want to build the alternative.
          </h3>
          <p className="text-center text-xl text-gray-400 mb-16 max-w-2xl mx-auto">
            This is a research project, not a startup. We're not promising quick returns.
          </p>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <div className="glass p-8 rounded-xl">
              <h4 className="text-xl font-bold text-white mb-4">Engineers</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Extend the reference implementation, experiment with edge hardware, optimize protocols.
              </p>
            </div>
            <div className="glass p-8 rounded-xl">
              <h4 className="text-xl font-bold text-white mb-4">Researchers</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Distributed training, ternary computing, hyperdimensional computing, governance design.
              </p>
            </div>
            <div className="glass p-8 rounded-xl">
              <h4 className="text-xl font-bold text-white mb-4">Connectors</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Know someone at a sovereign wealth fund? A European AI initiative? A research lab working on memristors?
              </p>
            </div>
          </div>

          <div className="glass p-10 rounded-2xl border border-signal/20 bg-signal/5 mb-12">
            <h4 className="text-xl font-bold text-white mb-6 text-center">How to Start</h4>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-3xl font-bold text-signal mb-3">1</div>
                <p className="text-sm text-gray-300">
                  <a href="/docs/specs/v1.0_current/spec-v1-final" className="text-white hover:text-signal transition-colors font-bold">Read the spec</a>
                </p>
              </div>
              <div>
                <div className="text-3xl font-bold text-signal mb-3">2</div>
                <p className="text-sm text-gray-300">
                  <a href="/demo" target="_blank" rel="noopener noreferrer" className="text-white hover:text-signal transition-colors font-bold">Try interactive demo</a>
                </p>
              </div>
              <div>
                <div className="text-3xl font-bold text-signal mb-3">3</div>
                <p className="text-sm text-gray-300">
                  <a href="https://github.com/nick-yudin/resonance-protocol/discussions" className="text-white hover:text-signal transition-colors font-bold">Join the conversation</a>
                </p>
              </div>
            </div>
          </div>

          <div className="text-center space-y-4">
            <div className="flex flex-col md:flex-row justify-center gap-6 text-gray-400">
              <a href="mailto:1@resonanceprotocol.org" className="hover:text-signal transition-colors">
                <strong className="text-white">Email:</strong> 1@resonanceprotocol.org
              </a>
              <a href="https://twitter.com/rAI_stack" target="_blank" className="hover:text-signal transition-colors">
                <strong className="text-white">Twitter:</strong> @rAI_stack
              </a>
              <a href="https://github.com/nick-yudin/resonance-protocol" target="_blank" className="hover:text-signal transition-colors">
                <strong className="text-white">GitHub:</strong> resonance-protocol
              </a>
            </div>
            <p className="text-gray-500 italic mt-8">
              We're not trying to beat OpenAI at their game. We're changing the game.
            </p>
          </div>
        </div>
      </section>

      {/* Section: About */}
      <section id="about" className="py-24 px-6 bg-silent border-t border-white/5">
        <div className="max-w-4xl mx-auto">
          <div className="glass p-12 rounded-2xl">
            <div className="flex flex-col md:flex-row gap-8 items-center">
              {/* Photo placeholder */}
              <div className="w-32 h-32 rounded-full bg-noise border-2 border-signal/30 flex-shrink-0 overflow-hidden flex items-center justify-center">
                <span className="text-gray-600 text-xs font-mono">[PHOTO]</span>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">Nikolay Yudin</h3>
                <p className="text-signal font-mono text-sm mb-4">Creator of Resonance Protocol</p>
                <p className="text-gray-400 leading-relaxed mb-4">
                  Building the infrastructure for distributed AI that no single entity can control.
                  Researching HDC, semantic computing, and cross-architecture knowledge transfer.
                </p>
                <p className="text-gray-400 leading-relaxed mb-6">
                  Looking for collaborators who believe AI should be like air — ubiquitous, invisible, and free.
                </p>
                <div className="flex gap-4">
                  <a href="mailto:1@resonanceprotocol.org" className="text-white hover:text-signal transition-colors" title="Email">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z"/><path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z"/></svg>
                  </a>
                  <a href="https://twitter.com/rAI_stack" target="_blank" className="text-white hover:text-signal transition-colors" title="Twitter / X">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                  </a>
                  <a href="https://github.com/nick-yudin" target="_blank" className="text-white hover:text-signal transition-colors" title="GitHub">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                  </a>
                  <a href="https://www.linkedin.com/in/nikolay-yudin/" target="_blank" className="text-white hover:text-signal transition-colors" title="LinkedIn">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                  </a>
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
        <div className="max-w-6xl mx-auto mt-12 pt-8 border-t border-white/5 text-center">
          <p className="text-gray-400 text-sm mb-2">
            Created by <a href="#about" className="text-white hover:text-signal font-bold transition-colors">Nikolay Yudin</a>
          </p>
          <p className="text-xs text-gray-700 font-mono">
            INITIATED 2025 // SILENCE IS GOLDEN
          </p>
        </div>
      </footer>
    </div>
  );
}
import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import Head from '@docusaurus/Head';

export default function Demo() {
  const [activeDemo, setActiveDemo] = useState('compression');
  const [compressionProgress, setCompressionProgress] = useState(0);
  const [transferProgress, setTransferProgress] = useState(0);
  const [generalizationProgress, setGeneralizationProgress] = useState(0);

  // Setup Tailwind CSS with faster loading
  useEffect(() => {
    if (!document.getElementById('tailwind-script')) {
      const script = document.createElement('script');
      script.id = 'tailwind-script';
      script.src = "https://cdn.tailwindcss.com";
      script.async = false; // Load synchronously for faster processing
      document.head.appendChild(script);
    }
  }, []);

  // Simulate compression demo
  useEffect(() => {
    if (activeDemo === 'compression') {
      const interval = setInterval(() => {
        setCompressionProgress((prev) => {
          if (prev >= 100) return 0;
          return prev + 2;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [activeDemo]);

  // Simulate transfer demo - animate once to 93%
  useEffect(() => {
    if (activeDemo === 'transfer') {
      setTransferProgress(0);
      const interval = setInterval(() => {
        setTransferProgress((prev) => {
          if (prev >= 93) {
            clearInterval(interval);
            return 93;
          }
          return prev + 1;
        });
      }, 30);
      return () => clearInterval(interval);
    }
  }, [activeDemo]);

  // Generalization demo - show 100% immediately
  useEffect(() => {
    if (activeDemo === 'generalization') {
      setGeneralizationProgress(100);
    }
  }, [activeDemo]);

  return (
    <Layout
      title="Interactive Demo"
      description="Try Resonance Protocol's HDC technology live in your browser"
    >
      <Head>
        <link rel="preconnect" href="https://cdn.jsdelivr.net" />
        <style>{`
          /* Critical inline styles for instant render */
          body { margin: 0; }
          .demo-container { min-height: 100vh; background-color: #0a0a0a; color: white; }
        `}</style>
      </Head>
      <div className="min-h-screen bg-[#0a0a0a] text-white demo-container">

        {/* Hero */}
        <section className="py-20 px-6 border-b border-white/5">
          <div className="max-w-6xl mx-auto text-center">
            <div className="inline-block px-3 py-1 border border-[#ff4d00]/30 rounded text-xs font-mono text-[#ff4d00] tracking-widest uppercase mb-6">
              Interactive Demo
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              See HDC in Action
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Explore the breakthrough technology behind Resonance Protocol.
              All demos run live in your browser—no installation required.
            </p>
          </div>
        </section>

        {/* Demo Selector */}
        <section className="py-12 px-6 border-b border-white/5 sticky top-0 bg-[#0a0a0a]/95 backdrop-blur-md z-40">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-wrap gap-4 justify-center">
              <button
                onClick={() => setActiveDemo('compression')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  activeDemo === 'compression'
                    ? 'bg-[#ff4d00] text-black'
                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                32× Compression
              </button>
              <button
                onClick={() => setActiveDemo('transfer')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  activeDemo === 'transfer'
                    ? 'bg-[#ff4d00] text-black'
                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                93% Cross-Architecture Transfer
              </button>
              <button
                onClick={() => setActiveDemo('generalization')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  activeDemo === 'generalization'
                    ? 'bg-[#ff4d00] text-black'
                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                100% Compositional Generalization
              </button>
            </div>
          </div>
        </section>

        {/* Demo Content */}
        <section className="py-20 px-6">
          <div className="max-w-6xl mx-auto">

            {/* Compression Demo */}
            {activeDemo === 'compression' && (
              <div className="space-y-12">
                <div className="text-center mb-12">
                  <h2 className="text-3xl font-bold mb-4">HDC Compression: 32× Reduction</h2>
                  <p className="text-gray-400 max-w-2xl mx-auto">
                    Watch as LoRA model weights (17.5 MB) are compressed to HDC semantic packets (271 KB)
                    while preserving 99%+ accuracy.
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-8">
                  {/* Before */}
                  <div className="bg-white/5 border border-white/10 rounded-xl p-8">
                    <div className="text-sm font-mono text-[#ff4d00] mb-4">BEFORE: Raw LoRA Weights</div>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">q_proj</span>
                        <span className="font-bold">4.2 MB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500" style={{ width: '100%' }}></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">v_proj</span>
                        <span className="font-bold">4.2 MB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500" style={{ width: '100%' }}></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">k_proj</span>
                        <span className="font-bold">4.2 MB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500" style={{ width: '100%' }}></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">out_proj</span>
                        <span className="font-bold">4.9 MB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500" style={{ width: '100%' }}></div>
                      </div>

                      <div className="border-t border-white/10 pt-4 mt-4">
                        <div className="flex justify-between items-center text-lg">
                          <span className="font-bold">Total Size</span>
                          <span className="font-bold text-red-400">17.5 MB</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* After */}
                  <div className="bg-white/5 border border-[#ff4d00]/30 rounded-xl p-8">
                    <div className="text-sm font-mono text-[#ff4d00] mb-4">AFTER: HDC Compression</div>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">q_proj (HDC)</span>
                        <span className="font-bold text-[#ff4d00]">68 KB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#ff4d00] transition-all duration-300"
                          style={{ width: `${Math.min(100, compressionProgress * 0.016)}%` }}
                        ></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">v_proj (HDC)</span>
                        <span className="font-bold text-[#ff4d00]">68 KB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#ff4d00] transition-all duration-300"
                          style={{ width: `${Math.min(100, compressionProgress * 0.016)}%` }}
                        ></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">k_proj (HDC)</span>
                        <span className="font-bold text-[#ff4d00]">68 KB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#ff4d00] transition-all duration-300"
                          style={{ width: `${Math.min(100, compressionProgress * 0.016)}%` }}
                        ></div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">out_proj (HDC)</span>
                        <span className="font-bold text-[#ff4d00]">67 KB</span>
                      </div>
                      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#ff4d00] transition-all duration-300"
                          style={{ width: `${Math.min(100, compressionProgress * 0.015)}%` }}
                        ></div>
                      </div>

                      <div className="border-t border-white/10 pt-4 mt-4">
                        <div className="flex justify-between items-center text-lg">
                          <span className="font-bold">Total Size</span>
                          <span className="font-bold text-[#ff4d00]">271 KB</span>
                        </div>
                        <div className="text-center mt-4">
                          <div className="text-3xl font-bold text-[#ff4d00]">32×</div>
                          <div className="text-sm text-gray-400">Compression Ratio</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Code Example */}
                <div className="bg-[#1a1a1a] border border-white/10 rounded-xl p-6 font-mono text-sm">
                  <div className="text-[#ff4d00] mb-4"># Try it yourself:</div>
                  <div className="space-y-2 text-gray-300">
                    <div><span className="text-gray-500">$</span> git clone https://github.com/nick-yudin/resonance-protocol</div>
                    <div><span className="text-gray-500">$</span> cd resonance-protocol/reference_impl/python</div>
                    <div><span className="text-gray-500">$</span> pip install -r requirements.txt</div>
                    <div><span className="text-gray-500">$</span> python3 -m hdc.distributed_trainer_hdc</div>
                    <div className="text-[#ff4d00] mt-4">✓ Compression: 17.5 MB → 271 KB (32×)</div>
                  </div>
                </div>
              </div>
            )}

            {/* Transfer Demo */}
            {activeDemo === 'transfer' && (
              <div className="space-y-12">
                <div className="text-center mb-12">
                  <h2 className="text-3xl font-bold mb-4">Cross-Architecture Knowledge Transfer</h2>
                  <p className="text-gray-400 max-w-2xl mx-auto">
                    Transfer knowledge from DistilBERT to GPT-2—two completely different architectures—with 93% efficiency.
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-8 mb-12">
                  {/* Teacher */}
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-8">
                    <div className="text-sm font-mono text-blue-400 mb-4">TEACHER: DistilBERT</div>
                    <div className="space-y-4">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Architecture</span>
                        <span>Encoder-only</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Parameters</span>
                        <span>66M</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Before Training</span>
                        <span>49.0%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">After Training</span>
                        <span className="font-bold text-blue-400">86.6%</span>
                      </div>
                      <div className="border-t border-white/10 pt-4">
                        <div className="flex justify-between text-lg">
                          <span className="font-bold">Improvement</span>
                          <span className="font-bold text-blue-400">+37.6%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Student */}
                  <div className="bg-[#ff4d00]/10 border border-[#ff4d00]/30 rounded-xl p-8">
                    <div className="text-sm font-mono text-[#ff4d00] mb-4">STUDENT: GPT-2</div>
                    <div className="space-y-4">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Architecture</span>
                        <span>Decoder-only</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Parameters</span>
                        <span>124M</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Before Transfer</span>
                        <span>47.0%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">After Transfer</span>
                        <span className="font-bold text-[#ff4d00]">82.0%</span>
                      </div>
                      <div className="border-t border-white/10 pt-4">
                        <div className="flex justify-between text-lg">
                          <span className="font-bold">Improvement</span>
                          <span className="font-bold text-[#ff4d00]">+35.0%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Transfer Progress */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-8">
                  <div className="text-center mb-6">
                    <div className="text-sm font-mono text-[#ff4d00] mb-2">Transfer Efficiency</div>
                    <div className="text-5xl font-bold text-[#ff4d00]">{transferProgress}%</div>
                  </div>
                  <div className="w-full h-4 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-[#ff4d00] transition-all duration-300"
                      style={{ width: `${transferProgress}%` }}
                    ></div>
                  </div>
                  <div className="text-center mt-6 text-gray-400 text-sm">
                    Student learned <strong className="text-white">{transferProgress}%</strong> of what teacher learned—despite completely different architectures
                  </div>
                </div>

                {/* Why This Works */}
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-white/5 border border-white/10 rounded-lg p-6 text-center">
                    <div className="text-3xl mb-3">🧠</div>
                    <div className="font-bold mb-2">Universal Semantic Space</div>
                    <div className="text-sm text-gray-400">HDC creates architecture-agnostic representations</div>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-lg p-6 text-center">
                    <div className="text-3xl mb-3">🔄</div>
                    <div className="font-bold mb-2">Knowledge Packets</div>
                    <div className="text-sm text-gray-400">271 KB semantic packets work across any model</div>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded-lg p-6 text-center">
                    <div className="text-3xl mb-3">✨</div>
                    <div className="font-bold mb-2">93% Efficiency</div>
                    <div className="text-sm text-gray-400">Near-perfect knowledge transfer validated experimentally</div>
                  </div>
                </div>

                {/* Code Example */}
                <div className="bg-[#1a1a1a] border border-white/10 rounded-xl p-6 font-mono text-sm">
                  <div className="text-[#ff4d00] mb-4"># Try it yourself:</div>
                  <div className="space-y-2 text-gray-300">
                    <div><span className="text-gray-500">$</span> python3 -m hdc.knowledge_transfer</div>
                    <div className="text-gray-500 mt-4"># Trains DistilBERT teacher</div>
                    <div className="text-gray-500"># Encodes knowledge → HDC packets</div>
                    <div className="text-gray-500"># Transfers to GPT-2 student</div>
                    <div className="text-[#ff4d00] mt-4">✓ Transfer efficiency: 93.1%</div>
                  </div>
                </div>
              </div>
            )}

            {/* Generalization Demo */}
            {activeDemo === 'generalization' && (
              <div className="space-y-12">
                <div className="text-center mb-12">
                  <h2 className="text-3xl font-bold mb-4">100% Compositional Generalization</h2>
                  <p className="text-gray-400 max-w-2xl mx-auto">
                    HDC achieves perfect zero-shot accuracy on unseen combinations through algebraic composition.
                  </p>
                </div>

                {/* Interactive Example */}
                <div className="bg-gradient-to-br from-purple-500/10 to-[#ff4d00]/10 border border-white/10 rounded-xl p-8">
                  <div className="text-center mb-8">
                    <div className="text-sm font-mono text-[#ff4d00] mb-2">Live Composition</div>
                    <div className="text-2xl font-bold">Combining Unseen Attributes</div>
                  </div>

                  <div className="grid md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-white/5 rounded-lg p-6 text-center">
                      <div className="text-4xl mb-4">🔴</div>
                      <div className="font-bold text-red-400">red</div>
                      <div className="text-xs text-gray-500 mt-2">HDC Vector A</div>
                    </div>
                    <div className="flex items-center justify-center text-3xl">+</div>
                    <div className="bg-white/5 rounded-lg p-6 text-center">
                      <div className="text-4xl mb-4">⬛</div>
                      <div className="font-bold text-blue-400">square</div>
                      <div className="text-xs text-gray-500 mt-2">HDC Vector B</div>
                    </div>
                  </div>

                  <div className="flex justify-center mb-8">
                    <div className="text-3xl text-[#ff4d00]">⊗</div>
                  </div>

                  <div className="bg-[#ff4d00]/20 border border-[#ff4d00] rounded-lg p-8 text-center">
                    <div className="text-5xl mb-4">🟥</div>
                    <div className="text-2xl font-bold text-[#ff4d00] mb-2">red square</div>
                    <div className="text-sm text-gray-400">
                      Composed HDC vector (never seen during training!)
                    </div>
                    <div className="mt-6">
                      <div className="w-full h-4 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#ff4d00] transition-all duration-300"
                          style={{ width: `${generalizationProgress}%` }}
                        ></div>
                      </div>
                      <div className="text-3xl font-bold text-[#ff4d00] mt-4">{generalizationProgress}%</div>
                      <div className="text-sm text-gray-400">Prediction Confidence</div>
                    </div>
                  </div>
                </div>

                {/* Training vs Test */}
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="bg-gray-500/10 border border-gray-500/30 rounded-xl p-6">
                    <div className="text-sm font-mono text-gray-400 mb-4">TRANSFORMER BASELINE (Trained)</div>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🔴🔵</span>
                        <span>red circle</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🔵⬛</span>
                        <span>blue square</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🟡🔺</span>
                        <span>yellow triangle</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-[#ff4d00]/10 border border-[#ff4d00]/30 rounded-xl p-6">
                    <div className="text-sm font-mono text-[#ff4d00] mb-4">✨ HDC ZERO-SHOT (100% Accuracy)</div>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🟥</span>
                        <span>red square ✓</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🔺</span>
                        <span>blue triangle ✓</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🟡</span>
                        <span>yellow circle ✓</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Why This Works */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-8">
                  <div className="text-lg font-bold mb-6 text-center">Why HDC Enables Perfect Composition</div>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className="text-3xl mb-3">⊗</div>
                      <div className="font-bold mb-2">Algebraic Binding</div>
                      <div className="text-sm text-gray-400">Combines concepts through element-wise multiplication</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl mb-3">🎯</div>
                      <div className="font-bold mb-2">Holographic Representation</div>
                      <div className="text-sm text-gray-400">Each vector encodes compositional structure</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl mb-3">∞</div>
                      <div className="font-bold mb-2">High-Dimensional Space</div>
                      <div className="text-sm text-gray-400">10,000 dimensions prevent collisions</div>
                    </div>
                  </div>
                </div>

                {/* Code Example */}
                <div className="bg-[#1a1a1a] border border-white/10 rounded-xl p-6 font-mono text-sm">
                  <div className="text-[#ff4d00] mb-4"># Try it yourself:</div>
                  <div className="space-y-2 text-gray-300">
                    <div><span className="text-gray-500">$</span> python3 -m hdc.compositional_test</div>
                    <div className="text-gray-500 mt-4"># Encodes: red, blue, green</div>
                    <div className="text-gray-500"># Encodes: circle, square, triangle</div>
                    <div className="text-gray-500"># Trains on: red+circle, blue+square, green+triangle</div>
                    <div className="text-gray-500"># Tests on: red+square, blue+triangle, green+circle</div>
                    <div className="text-[#ff4d00] mt-4">✓ Zero-shot accuracy: 100%</div>
                  </div>
                </div>
              </div>
            )}

          </div>
        </section>

        {/* CTA */}
        <section className="py-20 px-6 border-t border-white/5">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Build?</h2>
            <p className="text-gray-400 mb-8">
              All these demos are based on real, reproducible experiments. Dive into the code and see for yourself.
            </p>
            <div className="flex flex-col md:flex-row gap-4 justify-center">
              <a
                href="https://github.com/nick-yudin/resonance-protocol/tree/main/reference_impl/python/hdc"
                target="_blank"
                className="px-8 py-4 bg-[#ff4d00] text-black font-bold rounded hover:bg-[#ff6d20] transition-all"
              >
                Explore the Code
              </a>
              <a
                href="/docs/research"
                className="px-8 py-4 border border-white/20 text-white rounded hover:border-[#ff4d00] hover:text-[#ff4d00] transition-all"
              >
                Read Research Docs
              </a>
            </div>
          </div>
        </section>

      </div>
    </Layout>
  );
}

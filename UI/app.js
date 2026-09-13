const GRAPH_SERVICE_URL = "http://localhost:8003";
const DB_API_URL = "http://localhost:8000";

let currentConvId = null;
let conversations = JSON.parse(localStorage.getItem("graph_convs") || "[]");

const chatFeed = document.getElementById("chat-feed");
const userInput = document.getElementById("user-input");
const chatForm = document.getElementById("chat-form");
const convListEl = document.getElementById("conversations-list");
const activeConvIdEl = document.getElementById("active-conv-id");
const statusBadge = document.getElementById("status-badge");
const newChatBtn = document.getElementById("new-chat-btn");

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

newChatBtn.addEventListener("click", () => startNewChat());

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    userInput.value = "";
    appendUserMessage(query);

    const loaderId = appendLoadingMessage();

    try {
        const response = await fetch(`${GRAPH_SERVICE_URL}/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: query, conv_id: currentConvId })
        });

        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
        const data = await response.json();

        const normalizedData = normalizePayload(data);

        if (!currentConvId || currentConvId !== normalizedData.conv_id) {
            currentConvId = normalizedData.conv_id;
            saveConversation(currentConvId, query);
        }

        updateHeaderStatus(normalizedData.status);
        removeMessage(loaderId);
        appendAssistantMessage(normalizedData);

    } catch (err) {
        removeMessage(loaderId);
        appendErrorMessage(`Execution Error: ${err.message}`);
    }
});

function startNewChat() {
    currentConvId = null;
    activeConvIdEl.textContent = "conv_new";
    statusBadge.classList.add("hidden");
    chatFeed.innerHTML = `
        <div class="h-full flex flex-col items-center justify-center text-center mt-20 text-gray-500">
            <h2 class="text-2xl font-semibold text-gray-300 mb-2">Autonomous Graph Assistant</h2>
            <p class="text-sm max-w-md">Ask a question to generate execution graphs, resolve intent, and run validation nodes.</p>
        </div>
    `;
    renderSidebar();
}

function saveConversation(convId, initialPrompt) {
    if (!conversations.some(c => c.id === convId)) {
        conversations.unshift({ id: convId, title: initialPrompt.slice(0, 30) + "..." });
        localStorage.setItem("graph_convs", JSON.stringify(conversations));
        renderSidebar();
    }
}

async function deleteConversation(convId, event) {
    if (event) event.stopPropagation();

    try {
        const response = await fetch(`${DB_API_URL}/history/${encodeURIComponent(convId)}`, {
            method: "DELETE"
        });

        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);

        conversations = conversations.filter(c => c.id !== convId);
        localStorage.setItem("graph_convs", JSON.stringify(conversations));

        if (currentConvId === convId) {
            if (conversations.length > 0) {
                currentConvId = conversations[0].id;
                activeConvIdEl.textContent = currentConvId;
                loadConversationHistory(currentConvId);
            } else {
                startNewChat();
            }
        }

        renderSidebar();
    } catch (err) {
        console.error("[DELETE FAILED]:", err);
        alert(`Failed to delete conversation: ${err.message}`);
    }
}

async function loadConversationHistory(convId) {
    if (!convId || convId === "null" || convId === "undefined") {
        console.warn("[HISTORY] Fetch cancelled: convId is invalid:", convId);
        return;
    }

    const targetUrl = `${DB_API_URL}/history/${encodeURIComponent(convId)}`;
    chatFeed.innerHTML = '<div class="text-center text-gray-500 text-sm mt-10">Loading conversation history...</div>';
    
    try {
        const response = await fetch(targetUrl, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to reach history endpoint.`);
        
        const data = await response.json();
        chatFeed.innerHTML = "";

        const messages = data.messages || data.history || [];

        if (messages.length === 0) {
            chatFeed.innerHTML = `<div class="text-center text-gray-500 text-sm mt-10">No prior messages recorded for session <code class="text-cyan-400">${escapeHtml(convId)}</code></div>`;
            return;
        }

        messages.forEach(msg => {
            const role = String(msg.role || "").toLowerCase();

            if (role === "user") {
                const userText = typeof msg.message === "object" ? (msg.message.question || JSON.stringify(msg.message)) : msg.message;
                appendUserMessage(userText);
            } else if (role === "assistant") {
                let payload = {};
                
                if (typeof msg.message === "object" && msg.message !== null) {
                    payload = msg.message;
                } else if (typeof msg.message === "string") {
                    try {
                        const sanitized = msg.message.replace(/[\u0000-\u001F\u007F-\u009F]/g, (match) => {
                            if (match === "\n") return "\\n";
                            if (match === "\r") return "\\r";
                            if (match === "\t") return "\\t";
                            return "";
                        });
                        payload = JSON.parse(sanitized);
                    } catch (e) {
                        payload = { final_answer: msg.message, status: "SUCCESS" };
                    }
                }

                const normalizedPayload = normalizePayload(payload);
                appendAssistantMessage(normalizedPayload);
                if (normalizedPayload.status) updateHeaderStatus(normalizedPayload.status);
            }
        });

    } catch (err) {
        console.error("[HISTORY FETCH FAILED]:", err);
        chatFeed.innerHTML = `<div class="p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs font-mono">
            <strong>Fetch Error:</strong> ${escapeHtml(err.message)}<br/>
            <span class="text-gray-400">Target URL: ${escapeHtml(targetUrl)}</span>
        </div>`;
    }
}

function renderSidebar() {
    convListEl.innerHTML = "";
    conversations.forEach(c => {
        const item = document.createElement("div");
        item.className = `w-full px-3 py-2.5 rounded-xl text-xs flex items-center justify-between transition cursor-pointer group ${
            c.id === currentConvId ? "bg-[#2b2c2e] text-white" : "text-gray-400 hover:bg-[#1e1e1f]"
        }`;

        item.innerHTML = `
            <div class="flex flex-col min-w-0 pr-2 pointer-events-none">
                <span class="truncate font-medium">${escapeHtml(c.title)}</span>
                <span class="text-[10px] text-gray-500 font-mono">${c.id.slice(-4)}</span>
            </div>
            <button 
                onclick="deleteConversation('${c.id}', event)" 
                class="text-gray-500 hover:text-red-400 p-1 rounded opacity-0 group-hover:opacity-100 transition shrink-0" 
                title="Delete Chat"
            >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
            </button>
        `;

        item.onclick = () => {
            currentConvId = c.id;
            activeConvIdEl.textContent = currentConvId;
            renderSidebar();
            loadConversationHistory(c.id);
        };

        convListEl.appendChild(item);
    });
}

function updateHeaderStatus(status) {
    statusBadge.classList.remove("hidden", "bg-yellow-500/10", "text-yellow-400", "bg-emerald-500/10", "text-emerald-400");
    if (status === "NEEDS_CLARIFICATION" || status === "NEEDS_REVISION" || status === "CLARIFICATION_NEEDED") {
        statusBadge.textContent = status.replace("_", " ");
        statusBadge.classList.add("bg-yellow-500/10", "text-yellow-400", "border", "border-yellow-500/20");
    } else {
        statusBadge.textContent = "SUCCESS";
        statusBadge.classList.add("bg-emerald-500/10", "text-emerald-400", "border", "border-emerald-500/20");
    }
}

function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "flex justify-end";
    div.innerHTML = `<div class="bg-[#2b2c2e] text-gray-100 px-4 py-3 rounded-2xl max-w-2xl text-sm leading-relaxed">${escapeHtml(text)}</div>`;
    chatFeed.appendChild(div);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderTimelineHtml(timeline) {
    if (!Array.isArray(timeline) || timeline.length === 0) return "";

    return timeline.map((step, idx) => {
        const agent = step.agent || step.assigned_agent || step.node_type || "Execution Step";
        const status = step.status || step.action || "EXECUTED";
        const stepId = step.node_id || step.id || `step_${idx + 1}`;
        const duration = step.duration_seconds ? `${step.duration_seconds}s` : "";
        const preview = step.output_preview || step.reasoning || "";

        const badgeColor = (status === "EXECUTED" || status === "APPROVED" || status === "APPROVE")
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
            : (status === "NEEDS_CLARIFICATION" || status === "NEEDS_REVISION" || status === "CLARIFICATION_NEEDED")
            ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
            : "bg-blue-500/10 text-blue-400 border-blue-500/20";

        return `
            <div class="p-3 rounded-lg bg-[#111112] border border-gray-800/80 text-xs space-y-2">
                <div class="flex items-start gap-3">
                    <div class="w-6 h-6 rounded-full bg-blue-600/20 border border-blue-500/30 text-blue-400 flex items-center justify-center font-mono font-semibold shrink-0 text-[11px]">
                        ${idx + 1}
                    </div>
                    <div class="flex-1 min-w-0 space-y-1">
                        <div class="flex items-center justify-between gap-2">
                            <span class="font-semibold text-gray-200 truncate">${escapeHtml(agent)}</span>
                            <div class="flex items-center gap-2 shrink-0">
                                ${duration ? `<span class="text-[10px] text-gray-500 font-mono">${duration}</span>` : ""}
                                <span class="px-1.5 py-0.5 rounded text-[10px] font-mono border ${badgeColor}">${escapeHtml(status)}</span>
                            </div>
                        </div>
                        <div class="text-[10px] text-gray-500 font-mono">${escapeHtml(stepId)}</div>
                        ${preview ? `<p class="text-gray-300 font-mono text-[11px] leading-relaxed break-words mt-1">${escapeHtml(preview)}</p>` : ""}
                    </div>
                </div>

                <details class="mt-2 text-[11px] bg-[#0a0a0b] border border-gray-800/80 rounded-md overflow-hidden">
                    <summary class="px-2.5 py-1.5 cursor-pointer text-gray-400 hover:text-cyan-400 font-mono select-none flex justify-between items-center bg-[#151517]">
                        <span>View Step Payload & JSON Structure</span>
                        <span class="text-[10px] text-gray-500">▼</span>
                    </summary>
                    <div class="p-2.5 overflow-x-auto max-h-72 bg-[#0c0c0d]">
                        <pre class="text-cyan-300 font-mono text-[10px] leading-relaxed whitespace-pre-wrap break-all">${escapeHtml(JSON.stringify(step, null, 2))}</pre>
                    </div>
                </details>
            </div>
        `;
    }).join("");
}

function appendAssistantMessage(data) {
    const div = document.createElement("div");
    div.className = "flex gap-4 max-w-3xl";

    const answerText = data.final_answer || data.clarification_question || "";
    const parsedMarkdown = typeof marked !== "undefined" ? marked.parse(answerText) : escapeHtml(answerText);
    const timeline = data.execution_timeline || [];
    const imageUrl = data.graph_image_url;
    const mermaidCode = data.graph_mermaid;

    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center font-bold text-xs shrink-0">AG</div>
        <div class="flex-1 space-y-3 min-w-0">
            <div class="prose text-sm text-gray-200">${parsedMarkdown}</div>

            ${(imageUrl || mermaidCode) ? `
            <details class="text-xs bg-[#1e1e1f] border border-gray-800 rounded-xl overflow-hidden my-3">
                <summary class="px-3.5 py-2.5 cursor-pointer text-gray-300 hover:text-white font-mono select-none flex justify-between items-center bg-[#181819]">
                    <span class="flex items-center gap-2 font-semibold">
                        <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                        Architecture Graph Diagram
                    </span>
                    <span class="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/50 px-2 py-0.5 rounded font-mono">LANGGRAPH</span>
                </summary>
                
                <div class="p-4 bg-[#111112] border-t border-gray-800 flex flex-col items-center gap-3">
                    ${imageUrl ? `
                    <div class="w-full flex flex-col items-center gap-3 bg-[#0d0d0e] p-4 rounded-lg border border-gray-800/80">
                        <div class="w-full flex justify-end">
                            <button onclick="downloadGraphImage('${imageUrl}')" class="px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/40 border border-cyan-500/40 text-cyan-300 rounded-md text-[11px] font-mono flex items-center gap-1.5 transition cursor-pointer">
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                Download Graph PNG
                            </button>
                        </div>
                        <div class="w-full flex justify-center overflow-x-auto">
                            <img src="${imageUrl}" alt="LangGraph Architecture Diagram" class="max-h-96 object-contain rounded drop-shadow-md" loading="lazy" />
                        </div>
                    </div>
                    ` : ''}

                    ${mermaidCode ? `
                    <details class="w-full text-[11px] bg-[#0a0a0b] border border-gray-800/80 rounded-md overflow-hidden">
                        <summary class="px-2.5 py-1.5 cursor-pointer text-gray-400 hover:text-cyan-400 font-mono select-none flex justify-between items-center bg-[#151517]">
                            <span>Raw Mermaid Code</span>
                            <span class="text-[10px] text-gray-500">▼</span>
                        </summary>
                        <div class="p-2.5 overflow-x-auto max-h-48 bg-[#0c0c0d]">
                            <pre class="text-cyan-300 font-mono text-[10px] leading-relaxed whitespace-pre-wrap break-all">${escapeHtml(mermaidCode)}</pre>
                        </div>
                    </details>
                    ` : ''}
                </div>
            </details>
            ` : ''}

            ${timeline.length > 0 ? `
            <details class="text-xs bg-[#1e1e1f] border border-gray-800 rounded-xl overflow-hidden">
                <summary class="px-3.5 py-2.5 cursor-pointer text-gray-300 hover:text-white font-mono select-none flex justify-between items-center bg-[#181819]">
                    <span>Execution Timeline (${timeline.length} Steps)</span>
                    <span class="text-cyan-400 font-semibold">${data.status || 'SUCCESS'}</span>
                </summary>
                <div class="p-3 border-t border-gray-800 space-y-2 bg-[#171718]">
                    ${data.clarification_reasoning ? `<p class="text-yellow-400 text-xs mb-2"><strong>Reasoning:</strong> ${escapeHtml(data.clarification_reasoning)}</p>` : ''}
                    ${renderTimelineHtml(timeline)}
                </div>
            </details>
            ` : ''}
        </div>
    `;
    chatFeed.appendChild(div);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

async function downloadGraphImage(url) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = `langgraph-architecture-${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
    } catch (err) {
        window.open(url, "_blank");
    }
}

function appendLoadingMessage() {
    const id = "loader-" + Date.now();
    const div = document.createElement("div");
    div.id = id;
    div.className = "flex gap-4";
    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center font-bold text-xs shrink-0">AG</div>
        <div class="flex items-center gap-2 text-sm text-gray-400">
            <div class="w-2 h-2 rounded-full bg-blue-400 animate-ping"></div>
            Planner evaluating execution graph...
        </div>
    `;
    chatFeed.appendChild(div);
    chatFeed.scrollTop = chatFeed.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendErrorMessage(msg) {
    const div = document.createElement("div");
    div.className = "p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs font-mono";
    div.textContent = msg;
    chatFeed.appendChild(div);
}

function normalizePayload(data) {
    if (!data) return {};
    if (typeof data === "string") {
        try { data = JSON.parse(data); } catch (e) {}
    }
    if (!Array.isArray(data.execution_timeline)) {
        data.execution_timeline = [];
    }
    
    data.graph_mermaid = data.graph_mermaid || data.mermaid || data.mermaid_code || null;
    data.graph_image_url = data.graph_image_url || data.image_url || null;

    if (!data.graph_mermaid && data.execution_timeline.length > 0) {
        let mermaidLines = ["graph TD", "  Start([Start])"];
        data.execution_timeline.forEach((step, idx) => {
            const agent = (step.agent || step.assigned_agent || step.node_type || `Step ${idx + 1}`).replace(/[^a-zA-Z0-9_]/g, "_");
            const prev = idx === 0 ? "Start" : (data.execution_timeline[idx - 1].agent || `Step_${idx}`).replace(/[^a-zA-Z0-9_]/g, "_");
            mermaidLines.push(`  ${prev} --> ${agent}`);
        });
        const lastStep = data.execution_timeline[data.execution_timeline.length - 1];
        const lastAgent = (lastStep.agent || `Step_${data.execution_timeline.length}`).replace(/[^a-zA-Z0-9_]/g, "_");
        mermaidLines.push(`  ${lastAgent} --> End([End])`);
        data.graph_mermaid = mermaidLines.join("\n");
    }

    if (!data.graph_image_url && data.graph_mermaid) {
        try {
            const cleanMermaid = data.graph_mermaid.trim();
            const encoded = btoa(unescape(encodeURIComponent(cleanMermaid)));
            data.graph_image_url = `https://mermaid.ink/img/${encoded}`;
        } catch (e) {
            console.error("Failed to construct fallback image URL:", e);
        }
    }
    return data;
}

function escapeHtml(str) {
    if (typeof str !== "string") str = JSON.stringify(str) || "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function init() {
    renderSidebar();
    if (conversations.length > 0) {
        currentConvId = conversations[0].id;
        activeConvIdEl.textContent = currentConvId;
        loadConversationHistory(currentConvId);
    } else {
        startNewChat();
    }
}

init();
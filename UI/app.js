const GRAPH_SERVICE_URL = "http://localhost:8003";

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

async function loadConversationHistory(convId) {
    chatFeed.innerHTML = '<div class="text-center text-gray-500 text-sm mt-10">Loading conversation history...</div>';
    
    try {
        const response = await fetch(`${GRAPH_SERVICE_URL}/history/${convId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}: Failed to reach history endpoint.`);
        
        const data = await response.json();
        chatFeed.innerHTML = "";

        const messages = data.messages || [];

        if (messages.length === 0) {
            chatFeed.innerHTML = `<div class="text-center text-gray-500 text-sm mt-10">No prior messages recorded for session <code class="text-cyan-400">${convId}</code></div>`;
            return;
        }

        messages.forEach(msg => {
            const role = String(msg.role || "").toLowerCase();

            if (role === "user") {
                const userText = typeof msg.message === "object" ? (msg.message.question || JSON.stringify(msg.message)) : msg.message;
                appendUserMessage(userText);
            } else if (role === "assistant") {
                let payload = {};
                
                try {
                    payload = typeof msg.message === "string" ? JSON.parse(msg.message) : msg.message;
                } catch (e) {
                    payload = { final_answer: msg.message, status: "SUCCESS" };
                }

                const normalizedPayload = normalizePayload(payload);
                appendAssistantMessage(normalizedPayload);
                if (normalizedPayload.status) updateHeaderStatus(normalizedPayload.status);
            }
        });

    } catch (err) {
        console.error("[HISTORY RENDER ERROR]:", err);
        chatFeed.innerHTML = `<div class="p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs font-mono">Error: ${err.message}</div>`;
    }
}

function renderSidebar() {
    convListEl.innerHTML = "";
    conversations.forEach(c => {
        const item = document.createElement("button");
        item.className = `w-full text-left px-3 py-2.5 rounded-xl text-xs flex items-center justify-between transition ${c.id === currentConvId ? "bg-[#2b2c2e] text-white" : "text-gray-400 hover:bg-[#1e1e1f]"}`;
        item.innerHTML = `<span class="truncate font-medium">${c.title}</span><span class="text-[10px] text-gray-500 font-mono">${c.id.slice(-4)}</span>`;
        
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

    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center font-bold text-xs shrink-0">AG</div>
        <div class="flex-1 space-y-3">
            <div class="prose text-sm text-gray-200">${parsedMarkdown}</div>
            
            ${timeline.length > 0 ? `
            <details class="text-xs bg-[#1e1e1f] border border-gray-800 rounded-xl overflow-hidden">
                <summary class="px-3 py-2 cursor-pointer text-gray-400 hover:text-gray-200 font-mono select-none flex justify-between items-center">
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
    if (!Array.isArray(data.execution_timeline)) {
        data.execution_timeline = [];
    }
    return data;
}

function escapeHtml(str) {
    if (typeof str !== "string") str = JSON.stringify(str) || "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

startNewChat();
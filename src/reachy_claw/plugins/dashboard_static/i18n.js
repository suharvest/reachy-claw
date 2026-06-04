// ── Dashboard i18n ──────────────────────────────────────────────────
// Tiny, build-free internationalization layer. Loaded FIRST (before
// app.js / settings.js / diary.js / ha_settings.js) as a classic script.
//
// Usage:
//   t('key')                      -> translated string (falls back to en, then key)
//   t('key', {name: 'Bob'})       -> "{name}" placeholders interpolated
//   I18N.setLang('zh' | 'en')     -> switch language, persist, re-render DOM
//   I18N.apply(root)              -> translate [data-i18n*] nodes under root
//
// Static markup:
//   <span data-i18n="key">          textContent
//   <input data-i18n-placeholder="key">
//   <button data-i18n-title="key">   title attribute
//   <div data-i18n-html="key">       innerHTML (use sparingly, trusted only)
//
// Dynamic strings produced in JS should call t(...) at render time, and
// components with live text should re-render on the 'i18n:changed' event.
"use strict";

(function () {
  const DICT = {
    en: {
      // ── Nav / page ──
      "nav.live": "LIVE",
      "nav.diary": "DIARY",
      "lang.toggle": "中文",            // label shows the OTHER language
      "lang.toggle.title": "Switch language",

      // ── Live dashboard ──
      "see.title": "What I see",
      "video.connecting": "Connecting to vision...",
      "mind.possessive": "Reachy's",
      "mind.mind": "Mind",
      "mind.says": "Says",
      "mind.translation": "Translation",
      "hear.title": "What I hear",
      "asr.listening": "Listening...",
      "emotionLabel.smile": "Smiling face",

      // ── Diary page nav ──
      "diary.older": "Older",
      "diary.newer": "Newer",
      "diary.loading": "Loading...",
      "diary.narrate": "Narrate",
      "diary.stop": "Stop",
      "diary.noDiaries": "No diaries",
      "diary.today": "{date} · Today",
      "diary.day": "Day {date}",
      "diary.defaultTitle": "Daily Diary",
      "diary.endOfDiary": "End of diary",
      "diary.noDiaryFor": "No diary for {date}",
      "diary.noDiariesYet": "No diaries yet",
      "diary.emptyHint": "Diaries are generated daily from interaction data",
      "diary.loadingShort": "Loading",
      // diary section labels
      "diary.section.summary": "Summary",
      "diary.section.mood_curve": "Emotional Journey",
      "diary.section.conversations": "Conversations",
      "diary.section.faces": "People & Smiles",
      "diary.section.thoughts": "Reflections",
      "diary.section.environment": "Environment",
      "diary.section.smile_wall": "Smile Wall",
      "diary.visitor": "Visitor",
      "diary.reachy": "Reachy Mini",
      "diary.stat.people": "People",
      "diary.stat.smiles": "Smiles",
      "diary.stat.peak": "Peak",
      "diary.stat.recognized": "Recognized",
      "diary.stat.totalFaces": "Total Faces",
      "diary.stat.peakHour": "Peak Hour",
      "diary.sensor.temperature": "Temperature",
      "diary.sensor.humidity": "Humidity",
      "diary.sensor.weather": "Weather",
      "diary.sensor.location": "Location",
      "diary.dataSummary": "Data Summary",
      "diary.peakLabel": "Peak: {value}",

      // ── Settings shell ──
      "settings.title": "Settings",
      "settings.tab.mode": "General",
      "settings.tab.faces": "Face Management",
      "settings.tab.detail": "Details",
      "settings.tab.prompt": "Prompt",
      "settings.tab.diary": "Diary",
      "settings.tab.ha": "HA Sensors",

      // ── General tab ──
      "mode.conversation.title": "Conversation",
      "mode.conversation.desc": "Normal dialogue mode. Robot responds to user speech with emotion tags driving expressions.",
      "mode.monologue.title": "Monologue",
      "mode.monologue.desc": "Inner monologue mode. Robot observes and narrates thoughts.",
      "mode.interpreter.title": "Interpreter",
      "mode.interpreter.desc": "Simultaneous interpretation. Translates speech in real time.",
      "mode.current": "Current: {mode}",
      "lang.source": "Source Language",
      "lang.target": "Target Language",
      "lang.chinese": "Chinese",
      "lang.english": "English",
      "lang.japanese": "Japanese",
      "lang.korean": "Korean",
      "lang.french": "French",
      "lang.german": "German",
      "lang.spanish": "Spanish",
      "toggle.vlm": "Camera Vision (VLM)",
      "toggle.vlm.title": "Toggle VLM on/off",
      "toggle.bargein": "Barge-in (Interrupts)",
      "toggle.bargein.title": "Toggle barge-in",
      "memory.title": "Memory",
      "memory.turns": "{n} turns",
      "memory.hint": "How many conversation turns to remember (0 = stateless)",
      "volume.title": "Volume",
      "volume.mute.title": "Mute",
      "rest.title": "Rest Window",
      "rest.state.title": "Current rest state",
      "rest.placeholder": "— —",
      "rest.enabled": "Enabled",
      "rest.enabled.title": "Enable scheduled rest",
      "rest.window": "Window",
      "rest.timezone": "Timezone",
      "rest.save": "Save",
      "rest.forceSleep": "Force Sleep",
      "rest.forceSleep.title": "Force enter rest now",
      "rest.forceWake": "Force Wake",
      "rest.forceWake.title": "Force exit rest now",
      "rest.followSchedule": "Follow Schedule",
      "rest.followSchedule.title": "Drop manual override; follow schedule",
      "rest.resting": "Resting",
      "rest.restingForced": "Resting (forced)",
      "rest.awake": "Awake",
      "rest.awakeForced": "Awake (forced)",
      "rest.forcedInto": "Forced into rest",
      "rest.forcedAwake": "Forced awake",
      "rest.following": "Following schedule",
      "services.title": "Services",
      "services.restartAll": "Restart All Services",

      // ── Faces tab ──
      "faces.registered": "Registered Faces",
      "faces.loading": "Loading...",
      "faces.liveEnroll": "Live Camera Enroll",
      "faces.name": "Name",
      "faces.enroll": "Enroll",
      "faces.uploadImage": "Upload Image",
      "faces.dropHint": "Drop images here or click to select",
      "faces.fileTypes": "JPG / PNG",
      "faces.uploadEnroll": "Upload & Enroll",
      "faces.backup": "Backup",
      "faces.exportZip": "Export ZIP",
      "faces.importZip": "Import ZIP",
      "faces.smileCaptures": "Smile Captures",
      "faces.storage": "Storage",
      "faces.count": "Count",
      "faces.downloadZip": "Download ZIP",
      "faces.clearAll": "Clear All",
      "faces.none": "No faces registered",
      "faces.delete": "Delete",
      "faces.deleted": "Deleted: {name}",
      "faces.deleteFailed": "Delete failed",
      "faces.enterName": "Enter a name",
      "faces.enrollFailed": "Enroll failed",
      "faces.enrolled": "Enrolled: {name}",
      "faces.enrolledImages": "Enrolled {ok}/{total} images",
      "faces.enrolledImagesFail": "Enrolled {ok}/{total} images ({fail} failed)",
      "faces.exportFailed": "Export failed",
      "faces.exportedFaces": "Exported faces.zip",
      "faces.imported": "Imported {n} faces",
      "faces.importFailed": "Import failed",
      "faces.clearConfirm": "Clear all smile capture photos? This cannot be undone.",
      "faces.capturesCleared": "Captures cleared",
      "faces.clearFailed": "Clear failed",
      "faces.exportedCaptures": "Exported captures",
      "faces.photos": "{n} photos",

      // ── Details tab ──
      "llm.title": "LLM",
      "llm.backend": "Backend",
      "llm.ollamaUrl": "Ollama URL",
      "llm.model": "Model",
      "llm.gatewayHost": "Gateway Host",
      "llm.gatewayPort": "Gateway Port",
      "llm.apply": "Apply Changes",
      "llm.applying": "Applying LLM settings...",
      "voice.title": "Voice",
      "voice.cloned": "Cloned Voice",
      "voice.select": "-- Select Voice --",
      "voice.clone.title": "Clone new voice",
      "voice.speakerId": "Speaker ID",
      "voice.pitch": "Pitch",
      "voice.speed": "Speed",
      "audio.title": "Audio Detection",
      "audio.vadThreshold": "VAD Threshold",
      "audio.energyThreshold": "Energy Threshold",
      "audio.hint": "Higher = less sensitive to background noise",
      "motor.title": "Motor",
      "motor.title.toggle": "Toggle motor on/off",
      "motor.sensitive.title": "Sensitive",
      "motor.sensitive.desc": "Fast, responsive tracking. Best for demos and close interaction.",
      "motor.moderate.title": "Moderate",
      "motor.moderate.desc": "Balanced speed and smoothness. Default for most scenarios.",
      "motor.smart.title": "Smart",
      "motor.smart.desc": "Adaptive tracking that learns movement patterns. Smoother, more natural.",
      "motor.status": "Motor: {preset}",
      "motor.statusDisabled": "Motor: sleep (disabled)",

      // ── Prompt tab ──
      "prompt.conversation": "Conversation Prompt",
      "prompt.monologue": "Monologue Prompt",
      "prompt.interpreter": "Interpreter Prompt",
      "prompt.diary": "Diary Prompt",
      "prompt.diary.placeholder": "System prompt for daily diary generation. Reference sensor keys, set tone, customize sections.",
      "prompt.resetDefault": "Reset Default",
      "prompt.save": "Save",
      "prompt.saved": "Prompt saved: {mode}",

      // ── Diary settings tab ──
      "diarySet.autoPublish": "Auto Publish",
      "diarySet.autoPublishDaily": "Auto publish daily",
      "diarySet.autoPublish.title": "Toggle auto publish",
      "diarySet.privacyLinter": "Privacy linter",
      "diarySet.privacyLinter.title": "Toggle privacy linter",
      "diarySet.siteRepo": "Site Repository",
      "diarySet.repoUrl": "Repo URL",
      "diarySet.diaryPath": "Diary Path",
      "diarySet.branch": "Branch",
      "diarySet.save": "Save Diary Settings",
      "diarySet.history": "History (last 14 days)",
      "diarySet.haScope": "HA Sensors in scope",
      "diarySet.haScopeHint": "These sensor entities will be available to the diary prompt. Configure them in the HA Sensors tab.",
      "diarySet.noRecords": "No diary records yet.",
      "diarySet.published": "✓ Published",
      "diarySet.unpublished": "⚠ Unpublished",
      "diarySet.missing": "— Missing",
      "diarySet.regenerate": "Regenerate",
      "diarySet.publish": "Publish",
      "diarySet.genPublish": "Generate + Publish",
      "diarySet.working": "Working...",
      "diarySet.failed": "Failed: {msg}",

      // ── HA tab ──
      "ha.howItWorks": "How it works",
      "ha.help.html": "Reachy pulls sensor history from Home Assistant once per day, when the diary is generated. Selected entities (weather, room temperature, motion sensors, etc.) become context for the LLM — the diary prompt decides how to weave them in.<ol><li>Paste your HA URL + a long-lived access token below.<br><span class=\"ha-help-note\">In HA: Profile → Security → Long-Lived Access Tokens → Create Token.</span></li><li>Click <b>Test Connection</b> to verify.</li><li>Click <b>Refresh List</b>, check the entities you want Reachy to know about, then <b>Save Selection</b>.</li><li>That's it — they show up in tonight's diary automatically.</li></ol>",
      "ha.connection": "Connection",
      "ha.url": "HA URL",
      "ha.token": "Access Token",
      "ha.token.placeholder": "paste long-lived token",
      "ha.token.toggle": "Show/Hide token",
      "ha.test": "Test Connection",
      "ha.entities": "Entities",
      "ha.refresh": "Refresh List",
      "ha.save": "Save Selection",
      "ha.entitiesHint": "Click \"Refresh List\" after configuring connection.",
      "ha.selectionCount": "{n} selected of {total}",
      "ha.connected": "Connected",
      "ha.error": "Error",
      "ha.testing": "Testing…",
      "ha.noEntities": "No entities returned by HA.",
      "ha.savedField": "Saved {field}",
      "ha.saveFailed": "Save failed: {msg}",
      "ha.savedEntities": "Saved {n} entities",
      "ha.savedEntitiesDot": "Saved {n} entities.",
      "ha.saving": "Saving…",
      "ha.loading": "Loading…",
      "ha.networkError": "Network error: {msg}",
      "ha.http": "HTTP {status}",

      // ── Voice clone modal ──
      "clone.title": "Clone Voice",
      "clone.voiceName": "Voice Name",
      "clone.record": "Record",
      "clone.upload": "Upload",
      "clone.startRecording": "Start Recording",
      "clone.stop": "Stop",
      "clone.selectAudio": "Select Audio File",
      "clone.submit": "Clone Voice",
      "clone.cloning": "Cloning...",
      "clone.enterVoiceName": "Enter a voice name",
      "clone.recordFirst": "Record or upload audio first",
      "clone.readFailed": "Failed to read audio",
      "clone.recordingSaved": "Recording saved ({kb} KB)",
      "clone.cloned": "Voice cloned: {name}",
      "clone.failed": "Clone failed: {error}",
      "clone.micHttps": "Microphone requires HTTPS. Use https:// or localhost.",
      "clone.micBlocked": "Microphone blocked. Click the lock icon in address bar to allow.",
      "clone.micNotFound": "No microphone found on this device.",
      "clone.micError": "Microphone error: {msg}",

      // ── Smile gallery ──
      "smile.loading": "Loading smiles...",
      "smile.empty": "No smiles captured yet",
      "smile.error": "Could not load smiles",
      "smile.collected": "smiles collected",
      "common.close": "Close",

      // ── Misc / restart ──
      "common.notConnected": "Not connected",
      "common.plusOne": "+1",
      "restart.confirm": "Restart all services? The dashboard will briefly disconnect.",
      "restart.sending": "Sending restart command...",
      "restart.restarting": "Restarting services...",
      "restart.restartingOne": "Restarting {container}...",
      "restart.done": "All services restarted.",
      "restart.reconnecting": "All services restarted. Reconnecting...",
      "restart.error": "Error: {error}",
      "restart.failed": "Restart failed",
      "common.unknown": "unknown",
      "common.loadFailed": "Load failed: {msg}",
      "common.saved": "Saved",
      "mode.changed": "Mode: {mode}",

      // ── Emotions (pill + diary) ──
      "emotion.neutral": "Neutral",
      "emotion.happy": "Happy",
      "emotion.sad": "Sad",
      "emotion.thinking": "Thinking",
      "emotion.surprised": "Surprised",
      "emotion.curious": "Curious",
      "emotion.excited": "Excited",
      "emotion.confused": "Confused",
      "emotion.angry": "Angry",
      "emotion.laugh": "Laughing",
      "emotion.fear": "Fearful",
      "emotion.listening": "Listening",
      "emotion.contemplative": "Contemplative",
    },

    zh: {
      // ── Nav / page ──
      "nav.live": "实时",
      "nav.diary": "日记",
      "lang.toggle": "EN",
      "lang.toggle.title": "切换语言",

      // ── Live dashboard ──
      "see.title": "我看到的",
      "video.connecting": "正在连接视觉…",
      "mind.possessive": "",
      "mind.mind": "想法",
      "mind.says": "我说了",
      "mind.translation": "翻译",
      "hear.title": "我听到的",
      "asr.listening": "聆听中…",
      "emotionLabel.smile": "微笑",

      // ── Diary page nav ──
      "diary.older": "更早",
      "diary.newer": "更新",
      "diary.loading": "加载中…",
      "diary.narrate": "朗读",
      "diary.stop": "停止",
      "diary.noDiaries": "暂无日记",
      "diary.today": "{date} · 今天",
      "diary.day": "{date}",
      "diary.defaultTitle": "每日日记",
      "diary.endOfDiary": "日记结束",
      "diary.noDiaryFor": "{date} 没有日记",
      "diary.noDiariesYet": "暂无日记",
      "diary.emptyHint": "日记每天根据互动数据自动生成",
      "diary.loadingShort": "加载中",
      "diary.section.summary": "概要",
      "diary.section.mood_curve": "情绪轨迹",
      "diary.section.conversations": "对话",
      "diary.section.faces": "人物与微笑",
      "diary.section.thoughts": "随想",
      "diary.section.environment": "环境",
      "diary.section.smile_wall": "微笑墙",
      "diary.visitor": "访客",
      "diary.reachy": "Reachy Mini",
      "diary.stat.people": "人数",
      "diary.stat.smiles": "微笑",
      "diary.stat.peak": "高峰",
      "diary.stat.recognized": "已识别",
      "diary.stat.totalFaces": "人脸总数",
      "diary.stat.peakHour": "高峰时段",
      "diary.sensor.temperature": "温度",
      "diary.sensor.humidity": "湿度",
      "diary.sensor.weather": "天气",
      "diary.sensor.location": "位置",
      "diary.dataSummary": "数据汇总",
      "diary.peakLabel": "峰值：{value}",

      // ── Settings shell ──
      "settings.title": "设置",
      "settings.tab.mode": "通用",
      "settings.tab.faces": "人脸管理",
      "settings.tab.detail": "详细",
      "settings.tab.prompt": "提示词",
      "settings.tab.diary": "日记",
      "settings.tab.ha": "HA 传感器",

      // ── General tab ──
      "mode.conversation.title": "对话",
      "mode.conversation.desc": "常规对话模式。机器人根据用户语音回应，并用情绪标签驱动表情。",
      "mode.monologue.title": "独白",
      "mode.monologue.desc": "内心独白模式。机器人观察周围并讲述自己的想法。",
      "mode.interpreter.title": "同传",
      "mode.interpreter.desc": "同声传译。实时翻译语音。",
      "mode.current": "当前：{mode}",
      "lang.source": "源语言",
      "lang.target": "目标语言",
      "lang.chinese": "中文",
      "lang.english": "英语",
      "lang.japanese": "日语",
      "lang.korean": "韩语",
      "lang.french": "法语",
      "lang.german": "德语",
      "lang.spanish": "西班牙语",
      "toggle.vlm": "摄像头视觉 (VLM)",
      "toggle.vlm.title": "开启/关闭 VLM",
      "toggle.bargein": "插话打断",
      "toggle.bargein.title": "开启/关闭插话打断",
      "memory.title": "记忆",
      "memory.turns": "{n} 轮",
      "memory.hint": "记住多少轮对话（0 = 无状态）",
      "volume.title": "音量",
      "volume.mute.title": "静音",
      "rest.title": "休息时段",
      "rest.state.title": "当前休息状态",
      "rest.placeholder": "— —",
      "rest.enabled": "启用",
      "rest.enabled.title": "启用定时休息",
      "rest.window": "时段",
      "rest.timezone": "时区",
      "rest.save": "保存",
      "rest.forceSleep": "强制休息",
      "rest.forceSleep.title": "立即进入休息",
      "rest.forceWake": "强制唤醒",
      "rest.forceWake.title": "立即退出休息",
      "rest.followSchedule": "跟随计划",
      "rest.followSchedule.title": "取消手动覆盖，跟随计划",
      "rest.resting": "休息中",
      "rest.restingForced": "休息中（强制）",
      "rest.awake": "唤醒",
      "rest.awakeForced": "唤醒（强制）",
      "rest.forcedInto": "已强制进入休息",
      "rest.forcedAwake": "已强制唤醒",
      "rest.following": "已跟随计划",
      "services.title": "服务",
      "services.restartAll": "重启所有服务",

      // ── Faces tab ──
      "faces.registered": "已注册人脸",
      "faces.loading": "加载中…",
      "faces.liveEnroll": "实时摄像头录入",
      "faces.name": "姓名",
      "faces.enroll": "录入",
      "faces.uploadImage": "上传图片",
      "faces.dropHint": "拖拽图片到此处或点击选择",
      "faces.fileTypes": "JPG / PNG",
      "faces.uploadEnroll": "上传并录入",
      "faces.backup": "备份",
      "faces.exportZip": "导出 ZIP",
      "faces.importZip": "导入 ZIP",
      "faces.smileCaptures": "微笑抓拍",
      "faces.storage": "存储",
      "faces.count": "数量",
      "faces.downloadZip": "下载 ZIP",
      "faces.clearAll": "全部清除",
      "faces.none": "暂无已注册人脸",
      "faces.delete": "删除",
      "faces.deleted": "已删除：{name}",
      "faces.deleteFailed": "删除失败",
      "faces.enterName": "请输入姓名",
      "faces.enrollFailed": "录入失败",
      "faces.enrolled": "已录入：{name}",
      "faces.enrolledImages": "已录入 {ok}/{total} 张图片",
      "faces.enrolledImagesFail": "已录入 {ok}/{total} 张图片（{fail} 张失败）",
      "faces.exportFailed": "导出失败",
      "faces.exportedFaces": "已导出 faces.zip",
      "faces.imported": "已导入 {n} 个人脸",
      "faces.importFailed": "导入失败",
      "faces.clearConfirm": "清除所有微笑抓拍照片？此操作不可撤销。",
      "faces.capturesCleared": "抓拍已清除",
      "faces.clearFailed": "清除失败",
      "faces.exportedCaptures": "抓拍已导出",
      "faces.photos": "{n} 张照片",

      // ── Details tab ──
      "llm.title": "大模型",
      "llm.backend": "后端",
      "llm.ollamaUrl": "Ollama 地址",
      "llm.model": "模型",
      "llm.gatewayHost": "网关主机",
      "llm.gatewayPort": "网关端口",
      "llm.apply": "应用更改",
      "llm.applying": "正在应用大模型设置…",
      "voice.title": "语音",
      "voice.cloned": "克隆音色",
      "voice.select": "-- 选择音色 --",
      "voice.clone.title": "克隆新音色",
      "voice.speakerId": "说话人 ID",
      "voice.pitch": "音调",
      "voice.speed": "语速",
      "audio.title": "音频检测",
      "audio.vadThreshold": "VAD 阈值",
      "audio.energyThreshold": "能量阈值",
      "audio.hint": "越高 = 对背景噪声越不敏感",
      "motor.title": "电机",
      "motor.title.toggle": "开启/关闭电机",
      "motor.sensitive.title": "灵敏",
      "motor.sensitive.desc": "快速、灵敏的跟踪。适合演示和近距离互动。",
      "motor.moderate.title": "适中",
      "motor.moderate.desc": "速度与平滑兼顾。适用于大多数场景的默认值。",
      "motor.smart.title": "智能",
      "motor.smart.desc": "自适应跟踪，学习运动模式。更平滑、更自然。",
      "motor.status": "电机：{preset}",
      "motor.statusDisabled": "电机：休眠（已禁用）",

      // ── Prompt tab ──
      "prompt.conversation": "对话提示词",
      "prompt.monologue": "独白提示词",
      "prompt.interpreter": "同传提示词",
      "prompt.diary": "日记提示词",
      "prompt.diary.placeholder": "用于生成每日日记的系统提示词。可引用传感器键、设定语气、自定义章节。",
      "prompt.resetDefault": "恢复默认",
      "prompt.save": "保存",
      "prompt.saved": "提示词已保存：{mode}",

      // ── Diary settings tab ──
      "diarySet.autoPublish": "自动发布",
      "diarySet.autoPublishDaily": "每日自动发布",
      "diarySet.autoPublish.title": "开启/关闭自动发布",
      "diarySet.privacyLinter": "隐私检查",
      "diarySet.privacyLinter.title": "开启/关闭隐私检查",
      "diarySet.siteRepo": "站点仓库",
      "diarySet.repoUrl": "仓库地址",
      "diarySet.diaryPath": "日记路径",
      "diarySet.branch": "分支",
      "diarySet.save": "保存日记设置",
      "diarySet.history": "历史（最近 14 天）",
      "diarySet.haScope": "纳入的 HA 传感器",
      "diarySet.haScopeHint": "这些传感器实体将提供给日记提示词。请在「HA 传感器」标签页中配置。",
      "diarySet.noRecords": "暂无日记记录。",
      "diarySet.published": "✓ 已发布",
      "diarySet.unpublished": "⚠ 未发布",
      "diarySet.missing": "— 缺失",
      "diarySet.regenerate": "重新生成",
      "diarySet.publish": "发布",
      "diarySet.genPublish": "生成并发布",
      "diarySet.working": "处理中…",
      "diarySet.failed": "失败：{msg}",

      // ── HA tab ──
      "ha.howItWorks": "工作原理",
      "ha.help.html": "Reachy 每天在生成日记时从 Home Assistant 拉取一次传感器历史。所选实体（天气、室温、运动传感器等）会成为大模型的上下文——由日记提示词决定如何融入。<ol><li>在下方粘贴你的 HA 地址 + 长期访问令牌。<br><span class=\"ha-help-note\">在 HA 中：个人资料 → 安全 → 长期访问令牌 → 创建令牌。</span></li><li>点击<b>测试连接</b>进行验证。</li><li>点击<b>刷新列表</b>，勾选你希望 Reachy 了解的实体，然后点击<b>保存选择</b>。</li><li>就这样——它们会自动出现在今晚的日记里。</li></ol>",
      "ha.connection": "连接",
      "ha.url": "HA 地址",
      "ha.token": "访问令牌",
      "ha.token.placeholder": "粘贴长期访问令牌",
      "ha.token.toggle": "显示/隐藏令牌",
      "ha.test": "测试连接",
      "ha.entities": "实体",
      "ha.refresh": "刷新列表",
      "ha.save": "保存选择",
      "ha.entitiesHint": "配置连接后点击「刷新列表」。",
      "ha.selectionCount": "已选 {n} / 共 {total}",
      "ha.connected": "已连接",
      "ha.error": "错误",
      "ha.testing": "测试中…",
      "ha.noEntities": "HA 未返回任何实体。",
      "ha.savedField": "已保存 {field}",
      "ha.saveFailed": "保存失败：{msg}",
      "ha.savedEntities": "已保存 {n} 个实体",
      "ha.savedEntitiesDot": "已保存 {n} 个实体。",
      "ha.saving": "保存中…",
      "ha.loading": "加载中…",
      "ha.networkError": "网络错误：{msg}",
      "ha.http": "HTTP {status}",

      // ── Voice clone modal ──
      "clone.title": "克隆音色",
      "clone.voiceName": "音色名称",
      "clone.record": "录制",
      "clone.upload": "上传",
      "clone.startRecording": "开始录制",
      "clone.stop": "停止",
      "clone.selectAudio": "选择音频文件",
      "clone.submit": "克隆音色",
      "clone.cloning": "克隆中…",
      "clone.enterVoiceName": "请输入音色名称",
      "clone.recordFirst": "请先录制或上传音频",
      "clone.readFailed": "读取音频失败",
      "clone.recordingSaved": "录音已保存（{kb} KB）",
      "clone.cloned": "音色已克隆：{name}",
      "clone.failed": "克隆失败：{error}",
      "clone.micHttps": "麦克风需要 HTTPS。请使用 https:// 或 localhost。",
      "clone.micBlocked": "麦克风被阻止。请点击地址栏的锁形图标以允许。",
      "clone.micNotFound": "此设备未找到麦克风。",
      "clone.micError": "麦克风错误：{msg}",

      // ── Smile gallery ──
      "smile.loading": "正在加载微笑…",
      "smile.empty": "暂无微笑抓拍",
      "smile.error": "无法加载微笑",
      "smile.collected": "次微笑收藏",
      "common.close": "关闭",

      // ── Misc / restart ──
      "common.notConnected": "未连接",
      "common.plusOne": "+1",
      "restart.confirm": "重启所有服务？面板将短暂断开连接。",
      "restart.sending": "正在发送重启指令…",
      "restart.restarting": "正在重启服务…",
      "restart.restartingOne": "正在重启 {container}…",
      "restart.done": "所有服务已重启。",
      "restart.reconnecting": "所有服务已重启。正在重新连接…",
      "restart.error": "错误：{error}",
      "restart.failed": "重启失败",
      "common.unknown": "未知",
      "common.loadFailed": "加载失败：{msg}",
      "common.saved": "已保存",
      "mode.changed": "模式：{mode}",

      // ── Emotions ──
      "emotion.neutral": "平静",
      "emotion.happy": "开心",
      "emotion.sad": "难过",
      "emotion.thinking": "思考",
      "emotion.surprised": "惊讶",
      "emotion.curious": "好奇",
      "emotion.excited": "兴奋",
      "emotion.confused": "困惑",
      "emotion.angry": "生气",
      "emotion.laugh": "大笑",
      "emotion.fear": "害怕",
      "emotion.listening": "聆听",
      "emotion.contemplative": "沉思",
    },
  };

  const SUPPORTED = Object.keys(DICT);

  function pickInitialLang() {
    const saved = localStorage.getItem("lang");
    if (saved && DICT[saved]) return saved;
    const nav = (navigator.language || "en").toLowerCase();
    return nav.startsWith("zh") ? "zh" : "en";
  }

  let lang = pickInitialLang();

  function t(key, vars) {
    let s = DICT[lang] && DICT[lang][key];
    if (s == null) s = DICT.en[key];
    if (s == null) s = key;
    if (vars) {
      for (const k in vars) {
        s = s.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
      }
    }
    return s;
  }

  function apply(root) {
    root = root || document;
    root.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.getAttribute("data-i18n"));
    });
    root.querySelectorAll("[data-i18n-html]").forEach((el) => {
      el.innerHTML = t(el.getAttribute("data-i18n-html"));
    });
    root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
    });
    root.querySelectorAll("[data-i18n-title]").forEach((el) => {
      el.setAttribute("title", t(el.getAttribute("data-i18n-title")));
    });
    if (root === document || root === document.documentElement) {
      document.documentElement.lang = lang;
    }
  }

  function setLang(l) {
    if (!DICT[l] || l === lang) {
      if (DICT[l] && l === lang) return;
      if (!DICT[l]) return;
    }
    lang = l;
    try { localStorage.setItem("lang", l); } catch (e) { /* ignore */ }
    apply(document);
    document.dispatchEvent(new CustomEvent("i18n:changed", { detail: { lang } }));
  }

  function getLang() { return lang; }

  window.I18N = { t, apply, setLang, getLang, supported: SUPPORTED, DICT };
  // Convenience global (the dashboard scripts are plain classic scripts).
  window.t = t;

  // Translate the initial static markup as soon as the DOM is ready.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => apply(document));
  } else {
    apply(document);
  }
})();

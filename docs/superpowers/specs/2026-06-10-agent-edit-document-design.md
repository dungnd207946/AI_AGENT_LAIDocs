# Thiết kế: Agent chỉnh sửa document (old/new + confirm)

**Ngày:** 2026-06-10
**Trạng thái:** Đã chốt thiết kế, chờ implement

## Mục tiêu

Cho phép agent chat (DeepAgents) **chỉnh sửa nội dung Markdown** của document đang mở
(thêm / sửa / xóa) khi user yêu cầu, qua cơ chế `old_string` → `new_string` giống tool
Edit của Claude. Mọi thay đổi phải được **user xác nhận trước khi ghi** (HITL), và phần
hiển thị document trên frontend **tự reload** sau khi ghi thành công.

Lý do khả thi: content đã được lưu local và đã có sẵn đường ghi đồng bộ 3 nơi
(file `.md`, cột `documents.content` trong SQLite, cột `tree_index`) trong
[update_document](../../../backend/api/documents.py) (dòng ~389). Tool sẽ tái sử dụng
đúng đường ghi đó.

## 4 quyết định thiết kế đã chốt

1. **HITL = two-tool conversational** (không dùng LangGraph `interrupt()`): tách thành
   `preview_edit` (chỉ đọc, trả diff) và `apply_edit` (ghi thật). Agent tự hỏi user xác
   nhận giữa hai bước. Không đụng tới interrupt/resume machinery.
2. **Một cặp tool old/new** cho cả 3 thao tác: sửa = `old → new`; xóa = `new` rỗng;
   thêm = `old` là mỏ neo, `new` = mỏ neo + nội dung mới (insert tương đối, không có
   "chèn vào dòng X" thuần).
3. **Normalized match + preview echo**: khớp `old_string` chịu được sai lệch khoảng
   trắng/xuống dòng/header của LLM; preview trả về đúng đoạn raw thật sẽ bị thay.
4. **Auto-reload** qua sentinel SSE `[EDITED]`: backend báo nhẹ, frontend re-fetch
   document.

## Luồng tổng quan

```
User: "Sửa đoạn X thành Y" (hoặc "xóa mục Z", "thêm ghi chú vào sau phần W")
  → Agent gọi preview_edit(old, new)   ← CHỈ ĐỌC, trả diff (đoạn raw thật → new), KHÔNG ghi
  → Agent hiển thị diff + hỏi "Bạn duyệt thay đổi này?"
User: "OK"
  → Agent gọi apply_edit(old, new)     ← GHI: file .md + documents.content + rebuild tree
  → Stream phát sentinel [EDITED] trước [DONE]
  → Frontend nhận [EDITED] → re-fetch getDocument(docId) → cập nhật preview/editor
```

Không có user xác nhận rõ ràng → agent KHÔNG được gọi `apply_edit`.

## Thành phần

### 1. Helper định vị `_locate_in_content` (mới, trong `agent.py`)

`_locate_in_content(content: str, old_string: str) -> tuple[int, int] | str`

Trả về `(start, end)` là offset trong **raw content** của đoạn cần thay, hoặc một chuỗi
thông báo lỗi (để tool trả thẳng cho agent đọc).

Thuật toán:

1. **Exact match trước:** đếm `content.count(old_string)`.
   - đúng 1 lần → trả `(start, end)` của chính nó.
   - > 1 lần → lỗi: `"Đoạn text xuất hiện N lần — cần thêm ngữ cảnh để xác định duy nhất."`
2. **Nếu exact = 0 → normalized match:** gộp mọi run khoảng trắng/xuống dòng thành 1 space
   ở cả `content` và `old_string`, đồng thời giữ **bản đồ offset** từ vị trí đã chuẩn hóa
   ngược về offset raw. Tìm `old_string` đã chuẩn hóa trong `content` đã chuẩn hóa.
   - 0 lần → lỗi: `"Không tìm thấy đoạn text trong tài liệu."`
   - > 1 lần → lỗi: `"Đoạn text (sau chuẩn hóa) không duy nhất — cần thêm ngữ cảnh."`
   - đúng 1 lần → dùng bản đồ offset truy ra `(start, end)` của **đoạn raw thật**.

**Lưu ý chống match nhầm:** vì luôn yêu cầu khớp **duy nhất**, đoạn quá ngắn/lặp lại sẽ
bị từ chối kèm thông báo yêu cầu thêm ngữ cảnh, thay vì sửa nhầm chỗ.

### 2. Tool `preview_edit` (mới, `@tool`, sync, chỉ đọc)

`preview_edit(old_string: str, new_string: str) -> str`

- Lấy `doc_id` từ `_tool_context_var` (giống `retrieve_context`).
- Load raw content qua `_get_document_content(doc_id)` (đã có sẵn).
- Gọi `_locate_in_content`. Nếu trả lỗi → trả thẳng chuỗi lỗi cho agent.
- Nếu OK → dựng diff dạng text gồm:
  - **đoạn raw THẬT** `content[start:end]` (đã bỏ header `[Section:]`, đúng khoảng trắng
    gốc) — đây là "preview echo".
  - `new_string`.
  - Vài dòng ngữ cảnh trước/sau (tùy chọn) để user dễ đối chiếu.
- **Không ghi gì cả.**

### 3. Tool `apply_edit` (mới, `@tool`, async)

`apply_edit(old_string: str, new_string: str) -> str`

- Lấy `doc_id`, load raw content.
- Gọi lại `_locate_in_content` (deterministic — cho cùng `(start, end)` như preview, vì
  content chưa đổi trong cùng phiên). Nếu lỗi → trả chuỗi lỗi.
- `new_content = content[:start] + new_string + content[end:]`.
- Gọi helper dùng chung `persist_document_content(doc_id, new_content)`.
- Set cờ `_edited_flag_var` (contextvar) = True để `chat.py` biết phát sentinel.
- Trả về thông báo thành công ngắn (vd `"Đã áp dụng thay đổi vào tài liệu."`).

### 4. Helper ghi đồng bộ `persist_document_content` (refactor)

Tách logic ghi inline trong [update_document](../../../backend/api/documents.py) thành một
hàm dùng chung để **API endpoint và agent tool đi chung một đường ghi** (tránh lệch 3 nơi
lưu). Vị trí đề xuất: một hàm async trong tầng service (vd `services/document_store.py`
hoặc cùng chỗ tiện tái dụng).

`async persist_document_content(doc_id: str, new_content: str) -> None`

1. Lấy `(content, meta)` qua `vault.get_document(doc_id)` để biết `folder` + `filename`.
2. `vault.save_document(folder, filename, new_content)` — ghi file `.md` (+ `.meta.json`).
3. `UPDATE documents SET content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?`.
4. `tree = await build_tree_index(new_content)` rồi
   `UPDATE documents SET tree_index=? WHERE id=?` — **inline (await)** để `retrieve_context`
   ngay sau đó nhất quán với content mới.

`update_document` trong API được sửa để gọi helper này (giữ nguyên hành vi hiện tại của
endpoint; phần title/filename của endpoint xử lý riêng như cũ).

### 5. Tín hiệu auto-reload (SSE)

- contextvar `_edited_flag_var` (mặc định False) — **reset đầu mỗi request**
  (trong `set_tool_context` hoặc đầu `_event_generator`).
- Trong [chat.py](../../../backend/api/chat.py) khối `finally`: nếu cờ bật →
  `yield "data: [EDITED]\n\n"` **trước** `yield "data: [DONE]\n\n"`.
- [streamChat](../../../src/lib/sidecar.ts) thêm nhận diện sentinel `[EDITED]` (cùng phong
  cách `[DONE]`/`[ERROR]` hiện có) → gọi callback tùy chọn mới `onEdited?()`.
- `ChatPanel` truyền `onEdited` = re-fetch `getDocument(docId)` rồi cập nhật state
  preview/editor để hiển thị nội dung mới.

### 6. Đăng ký tool + sửa SOUL prompt (`agent.py`)

- `tools=[retrieve_context, read_image, preview_edit, apply_edit]`.
- Sửa `DOCUMENT_SOUL_PROMPT`:
  - **Giữ** Rule 2 (no fabrication) và Rule 5 (filesystem tools chỉ thấy scratch/memories
    — chỉnh sửa document KHÔNG đi qua filesystem tools mà qua 2 tool chuyên dụng).
  - **Thêm** mục mới (nội dung mẫu, sẽ tinh chỉnh khi implement):

    > ## Editing the Document
    > Bạn CÓ THỂ chỉnh sửa tài liệu khi user yêu cầu rõ ràng, qua 2 tool:
    > 1. Luôn gọi `preview_edit` trước để dựng diff từ nội dung thật của tài liệu.
    > 2. Trình diff cho user và hỏi xác nhận. TUYỆT ĐỐI không tự áp dụng.
    > 3. Chỉ khi user đồng ý rõ ràng → gọi `apply_edit` với CHÍNH XÁC `old_string` đã preview.
    > - `old_string` lấy từ `retrieve_context`/`preview_edit`, KHÔNG kèm dòng tiêu đề
    >   `[Section: ...]`, và không bịa.
    > - Nếu `preview_edit` báo "không tìm thấy" / "không duy nhất" → gọi `retrieve_context`
    >   lấy thêm ngữ cảnh rồi thử lại với `old_string` dài/đặc trưng hơn.
    > - Xóa = `new_string` rỗng. Thêm = dùng một đoạn đang có làm mỏ neo và nối nội dung
    >   mới vào. Không bao giờ ghi đè ngoài phạm vi user yêu cầu.

## Xử lý lỗi & edge cases

- `old_string` không khớp / khớp nhiều → tool trả **chuỗi thông báo lỗi rõ ràng** để agent
  hỏi lại user hoặc lấy thêm ngữ cảnh; **không crash stream** (giống cách `read_image` trả
  lỗi dạng text).
- `build_tree_index` lỗi → vẫn giữ content đã ghi (file + DB), log cảnh báo, không rollback
  (nhất quán với fallback hiện có khi không có tree index).
- Content đổi giữa `preview_edit` và `apply_edit` (hiếm, cùng phiên chat): `apply_edit`
  định vị lại; nếu không còn khớp → trả lỗi để agent preview lại.
- `new_string` rỗng (xóa): hợp lệ.
- Đồng thời mở document trong editor frontend: ngoài phạm vi v1 ngoài việc auto-reload sau
  khi agent ghi; không có lock chống ghi đè đồng thời ở v1.

## Phạm vi v1 (YAGNI)

**Trong phạm vi:** một cặp `preview_edit`/`apply_edit` với old/new + normalized match;
confirm hội thoại; auto-reload qua `[EDITED]`; refactor `persist_document_content`; sửa
prompt.

**Ngoài phạm vi (có thể làm sau):** LangGraph `interrupt()` HITL chuẩn; fuzzy match
(similarity threshold); tool insert riêng (append-to-end / insert-after-heading); lock
chống xung đột với editor; undo/backup phiên bản; rebuild tree theo từng node thay vì
toàn bộ.

## Kế hoạch test

- **Unit `_locate_in_content`:** exact 1 lần; exact nhiều lần (lỗi duy nhất); normalized
  khớp khi lệch khoảng trắng/xuống dòng; normalized không tìm thấy; normalized không duy
  nhất; khớp khi old_string vô tình kèm header `[Section:]`.
- **Unit `persist_document_content`:** xác nhận đồng bộ cả 3 nơi (file `.md`,
  `documents.content`, `documents.tree_index`).
- **Manual:** luồng đầy đủ preview → confirm → apply trên document thật; kiểm tra
  frontend auto-reload sau `[EDITED]`; thử 3 thao tác sửa/xóa/thêm.

## File dự kiến đụng tới

| File | Thay đổi |
|------|----------|
| [backend/services/agent.py](../../../backend/services/agent.py) | Thêm `_locate_in_content`, `preview_edit`, `apply_edit`, `_edited_flag_var`; đăng ký tool; sửa SOUL prompt |
| backend/services/document_store.py (mới) hoặc tầng service | `persist_document_content` |
| [backend/api/documents.py](../../../backend/api/documents.py) | `update_document` gọi helper chung |
| [backend/api/chat.py](../../../backend/api/chat.py) | Reset cờ; phát sentinel `[EDITED]` |
| [src/lib/sidecar.ts](../../../src/lib/sidecar.ts) | `streamChat` nhận diện `[EDITED]`, thêm callback `onEdited` |
| src/components/ChatPanel (frontend) | Truyền `onEdited` = re-fetch document |

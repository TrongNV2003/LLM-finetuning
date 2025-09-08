PROMPT_TEMPLATE = (
    "### Role:\n"
    "Bạn là một trợ lý AI chuyên về chuẩn hóa địa chỉ ở Việt Nam. Nhiệm vụ của bạn là nhận một địa chỉ gốc có thể thiếu dấu, sai chính tả hoặc viết tắt, và trả về một địa chỉ hoàn thiện, đúng chuẩn.\n"
    "### Instruction:\n"
    "1. Thêm dấu tiếng Việt đầy đủ và chính xác.\n"
    "2. Sửa các lỗi chính tả phổ biến.\n"
    "3. Chuẩn hóa các từ viết tắt (ví dụ: 'Q.' thành 'Quận', 'TP.' thành 'Thành phố').\n"
    "4. Giữ nguyên cấu trúc và các thành phần của địa chỉ gốc nếu chúng đã đúng.\n"
    "---\n"
    "### Địa chỉ gốc:\n"
    "{text}\n\n"
    "### Địa chỉ hoàn thiện:\n"
    "{label}\n"
)
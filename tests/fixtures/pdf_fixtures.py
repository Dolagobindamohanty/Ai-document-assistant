from io import BytesIO


def generate_pdf_bytes(pages_text: list[str]) -> bytes:
    """
    Generate valid, parseable multi-page PDF bytes with exact cross-reference (xref)
    offsets and standard Helvetica font dictionary.
    """
    header = b"%PDF-1.4\r\n"
    body = bytearray()
    offsets: dict[int, int] = {}

    def add_obj(obj_id: int, content_bytes: bytes) -> None:
        offsets[obj_id] = len(header) + len(body)
        body.extend(f"{obj_id} 0 obj\r\n".encode("latin1"))
        body.extend(content_bytes)
        body.extend(b"\r\nendobj\r\n")

    # Object 1: Catalog
    add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")

    num_pages = len(pages_text)
    page_ids = [3 + i * 2 for i in range(num_pages)]
    content_ids = [4 + i * 2 for i in range(num_pages)]
    font_id = 3 + num_pages * 2

    # Object 2: Pages tree
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    add_obj(2, f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>".encode("latin1"))

    # Objects for Pages and Content streams
    for pid, cid, text in zip(page_ids, content_ids, pages_text):
        add_obj(
            pid,
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/MediaBox [0 0 612 792] /Contents {cid} 0 R >>"
            ).encode("latin1"),
        )
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT\r\n/F1 12 Tf\r\n72 712 Td\r\n({safe_text}) Tj\r\nET\r\n".encode("latin1")
        content_data = f"<< /Length {len(stream)} >>\r\nstream\r\n".encode("latin1") + stream + b"endstream"
        add_obj(cid, content_data)

    # Object for Font
    add_obj(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_offset = len(header) + len(body)
    total_objs = font_id + 1

    # Exact 20-byte standard xref entries
    xref = bytearray()
    xref.extend(f"xref\r\n0 {total_objs}\r\n".encode("latin1"))
    xref.extend(b"0000000000 65535 f \r\n")
    for i in range(1, total_objs):
        off = offsets[i]
        xref.extend(f"{off:010d} 00000 n \r\n".encode("latin1"))

    trailer = (
        f"trailer\r\n<< /Size {total_objs} /Root 1 0 R >>\r\n"
        f"startxref\r\n{xref_offset}\r\n%%EOF\r\n"
    ).encode("latin1")

    return bytes(header + body + xref + trailer)


def generate_corrupt_pdf_bytes() -> bytes:
    """Return invalid/corrupt non-PDF bytes."""
    return b"%PDF-1.4\r\nInvalid corrupt structure without endobj or trailer"


def generate_empty_bytes() -> bytes:
    """Return 0-byte stream."""
    return b""

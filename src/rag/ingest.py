"""Command-line ingestion of reviewed PDF documents into local ChromaDB."""

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from src.rag.service import add_document_chunks, split_text


def _xml_local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _jats_xml_chunks(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Fichier XML JATS invalide : {path.name}") from exc

    title = next(
        (
            " ".join(element.itertext()).strip()
            for element in root.iter()
            if _xml_local_name(element.tag) == "article-title"
        ),
        "",
    )
    journal = next(
        (
            " ".join(element.itertext()).strip()
            for element in root.iter()
            if _xml_local_name(element.tag) == "journal-title"
        ),
        "",
    )
    doi = next(
        (
            (element.text or "").strip()
            for element in root.iter()
            if _xml_local_name(element.tag) == "article-id"
            and element.attrib.get("pub-id-type") == "doi"
        ),
        "",
    )
    authors = []
    for group in root.iter():
        if _xml_local_name(group.tag) != "contrib-group":
            continue
        for contributor in group:
            if (
                _xml_local_name(contributor.tag) != "contrib"
                or contributor.attrib.get("contrib-type", "author") != "author"
            ):
                continue
            given = next(
                (
                    " ".join(child.itertext()).strip()
                    for child in contributor.iter()
                    if _xml_local_name(child.tag) == "given-names"
                ),
                "",
            )
            surname = next(
                (
                    " ".join(child.itertext()).strip()
                    for child in contributor.iter()
                    if _xml_local_name(child.tag) == "surname"
                ),
                "",
            )
            author = " ".join(part for part in (given, surname) if part)
            if author:
                authors.append(author)
    publication_date = ""
    for element in root.iter():
        if _xml_local_name(element.tag) != "pub-date":
            continue
        parts = {
            _xml_local_name(child.tag): (child.text or "").strip()
            for child in element
        }
        if parts.get("year"):
            publication_date = "-".join(
                [
                    parts["year"],
                    parts.get("month", "01").zfill(2),
                    parts.get("day", "01").zfill(2),
                ]
            )
            break

    language = root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
    body = next(
        (element for element in root.iter() if _xml_local_name(element.tag) == "body"),
        None,
    )
    if not title or body is None:
        raise ValueError(
            f"Le XML doit être un article JATS avec titre et corps : {path.name}"
        )

    def paragraphs_without_references(element):
        excluded = {
            "fig",
            "table-wrap",
            "ref-list",
            "supplementary-material",
            "ack",
            "funding-group",
        }
        tag = _xml_local_name(element.tag)
        if tag in excluded:
            return
        if tag == "p":
            text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
            if text:
                yield text
            return
        for child in element:
            yield from paragraphs_without_references(child)

    chunks = []
    section_names = []
    abstract = next(
        (element for element in root.iter() if _xml_local_name(element.tag) == "abstract"),
        None,
    )
    if abstract is not None:
        for text in paragraphs_without_references(abstract):
            for chunk in split_text(text):
                chunks.append(chunk)
                section_names.append("Abstract")

    sections = [
        element for element in body if _xml_local_name(element.tag) == "sec"
    ]
    if not sections:
        sections = [body]

    for section_index, section in enumerate(sections, 1):
        section_title = next(
            (
                " ".join(child.itertext()).strip()
                for child in section
                if _xml_local_name(child.tag) == "title"
            ),
            f"Section {section_index}",
        )
        for text in paragraphs_without_references(section):
            for chunk in split_text(text):
                chunks.append(chunk)
                section_names.append(section_title)

    if not chunks:
        raise ValueError(f"Aucun texte exploitable dans {path.name}.")

    return {
        "title": title,
        "journal": journal,
        "doi": doi,
        "authors": authors,
        "publication_date": publication_date,
        "language": language,
        "chunks": chunks,
        "sections": section_names,
    }


def _extract_page_text(page, pdf_path, page_number, use_ocr, ocr_document=None):
    text = page.extract_text() or ""
    if text.strip() or not use_ocr:
        return text, ocr_document

    try:
        import pypdfium2
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR requis pour une page sans texte. Installez pypdfium2 et "
            "pytesseract, puis Tesseract OCR sur le système."
        ) from exc

    if ocr_document is None:
        ocr_document = pypdfium2.PdfDocument(str(pdf_path))
    bitmap = ocr_document[page_number].render(scale=2)
    return pytesseract.image_to_string(bitmap.to_pil(), lang="fra+eng"), ocr_document


def ingest_document(
    document_path,
    language,
    source,
    document_type,
    trust_level,
    publication_date=None,
    last_reviewed=None,
    rights_evidence=None,
    reuse_license=None,
    reuse_license_url=None,
    use_ocr=True,
):
    path = Path(document_path)
    if not path.is_file() or path.suffix.lower() not in {".pdf", ".xml"}:
        raise ValueError(f"Le PDF ou XML JATS est introuvable ou invalide : {path}")
    if not isinstance(language, str) or language not in {"fr", "en"}:
        raise ValueError("language doit être 'fr' ou 'en'.")
    if not isinstance(trust_level, str) or trust_level not in {
        "unverified",
        "approved",
    }:
        raise ValueError("trust_level doit être 'unverified' ou 'approved'.")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(document_type, str)
        or not document_type.strip()
    ):
        raise ValueError("source et document_type sont obligatoires.")
    if trust_level == "approved" and not last_reviewed:
        raise ValueError(
            "last_reviewed est obligatoire pour approuver un document."
        )
    if trust_level == "approved" and (
        not isinstance(rights_evidence, str)
        or not rights_evidence.startswith("https://")
    ):
        raise ValueError(
            "rights_evidence doit être une URL HTTPS vérifiant la licence "
            "pour approuver un document."
        )
    if trust_level == "approved" and (
        not isinstance(reuse_license, str) or not reuse_license.strip()
    ):
        raise ValueError(
            "reuse_license est obligatoire pour approuver un document."
        )
    if trust_level == "approved" and (
        not isinstance(reuse_license_url, str)
        or not reuse_license_url.startswith("https://")
    ):
        raise ValueError(
            "reuse_license_url doit être une URL HTTPS pour approuver un document."
        )
    for field_name, field_value in (
        ("publication_date", publication_date),
        ("last_reviewed", last_reviewed),
    ):
        if field_value:
            try:
                date.fromisoformat(field_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} doit respecter le format AAAA-MM-JJ."
                ) from exc

    article = None
    reader = None
    if path.suffix.lower() == ".xml":
        article = _jats_xml_chunks(path)
        if article["language"] and article["language"][:2] != language:
            raise ValueError(
                f"La langue déclarée ({language}) ne correspond pas au XML "
                f"({article['language']})."
            )
        chunks = article["chunks"]
        chunk_sections = article["sections"]
    else:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError(
                f"Le PDF est chiffré et ne peut pas être indexé : {path.name}"
            )
        chunks = []
        chunk_sections = []
        ocr_document = None
        try:
            for page_number, page in enumerate(reader.pages, 1):
                text, ocr_document = _extract_page_text(
                    page, path, page_number - 1, use_ocr, ocr_document
                )
                text = re.sub(r"\s+", " ", text).strip()
                for chunk in split_text(text):
                    chunks.append(chunk)
                    chunk_sections.append(str(page_number))
        finally:
            if ocr_document is not None:
                ocr_document.close()

        if not chunks:
            raise ValueError(
                f"Aucun texte exploitable dans {path.name}. Vérifiez le PDF et l'OCR."
            )

    document_id = hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:20]
    metadata = {
        "document_id": document_id,
        "title": (
            article["title"]
            if article
            else (reader.metadata.title if reader.metadata else None)
        ) or path.stem,
        "source": source.strip(),
        "language": language,
        "document_type": document_type.strip(),
        "publication_date": (
            publication_date
            or (
                article["publication_date"]
                if article
                else str(reader.metadata.get("/PublicationDate", "") or "")
            )
        ),
        "journal": (
            article["journal"]
            if article
            else str(reader.metadata.get("/Journal", "") or "")
        ),
        "doi": (
            article["doi"]
            if article
            else str(reader.metadata.get("/DOI", "") or "")
        ),
        "authors": (
            ", ".join(article["authors"])
            if article
            else str(reader.metadata.author or "")
        ),
        "last_reviewed": last_reviewed or "",
        "reuse_license": reuse_license or "",
        "reuse_license_url": reuse_license_url or "",
        "rights_evidence": rights_evidence or "",
        "trust_level": trust_level,
        "file_name": path.name,
    }

    passage_metadata = [
        {
            **metadata,
            "page_or_section": str(page_number),
        }
        for page_number in chunk_sections
    ]
    return add_document_chunks(chunks, passage_metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Indexer un PDF ou XML JATS local dans ChromaDB RenYou."
    )
    parser.add_argument("document", help="Chemin d'un PDF ou XML JATS")
    parser.add_argument("--language", required=True, choices=["fr", "en"])
    parser.add_argument("--source", required=True, help="Organisme ou URL de référence")
    parser.add_argument("--document-type", required=True, help="Guide, article, fiche...")
    parser.add_argument(
        "--trust-level",
        choices=["unverified", "approved"],
        default="unverified",
        help="Seuls les documents approved seront utilisés en réponse.",
    )
    parser.add_argument("--publication-date")
    parser.add_argument("--last-reviewed")
    parser.add_argument(
        "--rights-evidence",
        help="URL HTTPS prouvant que l'utilisation prévue est autorisée.",
    )
    parser.add_argument(
        "--reuse-license",
        help="Nom de la licence vérifiée, par exemple CC BY 4.0.",
    )
    parser.add_argument(
        "--reuse-license-url",
        help="URL HTTPS de la licence vérifiée.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Ne pas tenter l'OCR des pages scannées.",
    )
    args = parser.parse_args()

    count = ingest_document(
        args.document,
        language=args.language,
        source=args.source,
        document_type=args.document_type,
        trust_level=args.trust_level,
        publication_date=args.publication_date,
        last_reviewed=args.last_reviewed,
        rights_evidence=args.rights_evidence,
        reuse_license=args.reuse_license,
        reuse_license_url=args.reuse_license_url,
        use_ocr=not args.no_ocr,
    )
    print(f"{count} passages indexés depuis {args.document}.")


if __name__ == "__main__":
    main()

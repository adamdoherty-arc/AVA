"""
PDF Parser - Document Understanding and Analysis
================================================

Parse and analyze PDF documents including earnings reports, research papers,
and financial statements.

Features:
- PDF text extraction
- Table extraction and parsing
- Chart/image extraction from PDFs
- Financial statement analysis
- Earnings report parsing
- Research paper summarization

Supported Libraries:
- PyPDF2: Basic PDF text extraction
- pdfplumber: Advanced table extraction
- PyMuPDF (fitz): High-quality text and image extraction
- Camelot: Specialized table extraction

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


# ============================================================================
# PDF Document Types
# ============================================================================

class DocumentType(Enum):
    """Types of PDF documents"""
    EARNINGS_REPORT = "earnings_report"
    FINANCIAL_STATEMENT = "financial_statement"
    RESEARCH_REPORT = "research_report"
    SEC_FILING = "sec_filing"
    PRESENTATION = "presentation"
    GENERAL = "general"


# ============================================================================
# PDF Parser
# ============================================================================

class PDFParser:
    """
    Parse and analyze PDF documents

    Supports multiple parsing backends with automatic fallback.
    """

    def __init__(self) -> None:
        """Initialize PDF parser"""
        self.backends_available = self._check_available_backends()
        logger.info(f"PDF parser initialized with backends: {list(self.backends_available.keys())}")

    def _check_available_backends(self) -> Dict[str, bool]:
        """Check which PDF parsing backends are available"""
        available = {}

        # PyPDF2
        try:
            import PyPDF2
            available['pypdf2'] = True
        except ImportError:
            available['pypdf2'] = False
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")

        # pdfplumber
        try:
            import pdfplumber
            available['pdfplumber'] = True
        except ImportError:
            available['pdfplumber'] = False
            logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")

        # PyMuPDF (fitz)
        try:
            import fitz
            available['pymupdf'] = True
        except ImportError:
            available['pymupdf'] = False
            logger.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")

        # Camelot (for tables)
        try:
            import camelot
            available['camelot'] = True
        except ImportError:
            available['camelot'] = False
            logger.info("Camelot not installed (optional). Install with: pip install camelot-py[cv]")

        if not any(available.values()):
            raise RuntimeError("No PDF parsing backends available. Install at least PyPDF2 or PyMuPDF.")

        return available

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        document_type: DocumentType = DocumentType.GENERAL,
        extract_tables: bool = True,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """
        Parse PDF document

        Args:
            pdf_path: Path to PDF file
            document_type: Type of document (for specialized parsing)
            extract_tables: Extract tables from PDF
            extract_images: Extract images from PDF

        Returns:
            Dict with parsed content
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Basic text extraction
        text = self._extract_text(pdf_path)

        # Metadata
        metadata = self._extract_metadata(pdf_path)

        # Tables (if requested)
        tables = []
        if extract_tables:
            tables = self._extract_tables(pdf_path)

        # Images (if requested)
        images = []
        if extract_images:
            images = self._extract_images(pdf_path)

        # Document-specific analysis
        analysis = self._analyze_document(text, tables, document_type)

        return {
            'status': 'success',
            'pdf_path': str(pdf_path),
            'document_type': document_type.value,
            'text': text,
            'metadata': metadata,
            'tables': tables,
            'images': images,
            'analysis': analysis,
            'page_count': metadata.get('page_count', 0)
        }

    def parse_earnings_report(
        self,
        pdf_path: Union[str, Path],
        ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse earnings report PDF with specialized extraction

        Args:
            pdf_path: Path to earnings report PDF
            ticker: Stock ticker (optional)

        Returns:
            Dict with earnings data
        """
        result = self.parse_pdf(
            pdf_path=pdf_path,
            document_type=DocumentType.EARNINGS_REPORT,
            extract_tables=True
        )

        # Extract earnings-specific data
        text = result['text']
        tables = result['tables']

        earnings_data = {
            'ticker': ticker,
            'revenue': self._extract_revenue(text, tables),
            'eps': self._extract_eps(text, tables),
            'guidance': self._extract_guidance(text),
            'key_metrics': self._extract_key_metrics(text, tables),
            'management_commentary': self._extract_management_commentary(text),
            'quarter': self._extract_quarter(text),
            'year': self._extract_year(text)
        }

        result['earnings_data'] = earnings_data

        return result

    def parse_financial_statement(
        self,
        pdf_path: Union[str, Path],
        statement_type: str = "income_statement"
    ) -> Dict[str, Any]:
        """
        Parse financial statement (income statement, balance sheet, cash flow)

        Args:
            pdf_path: Path to financial statement PDF
            statement_type: Type of statement

        Returns:
            Dict with financial data
        """
        result = self.parse_pdf(
            pdf_path=pdf_path,
            document_type=DocumentType.FINANCIAL_STATEMENT,
            extract_tables=True
        )

        # Extract financial statement data
        tables = result['tables']

        financial_data = {
            'statement_type': statement_type,
            'data': self._parse_financial_tables(tables, statement_type)
        }

        result['financial_data'] = financial_data

        return result

    # ========================================================================
    # Text Extraction Methods
    # ========================================================================

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using best available backend"""
        # Try PyMuPDF first (best quality)
        if self.backends_available.get('pymupdf'):
            try:
                return self._extract_text_pymupdf(pdf_path)
            except Exception as e:
                logger.error(f"PyMuPDF extraction failed: {e}")

        # Fallback to pdfplumber
        if self.backends_available.get('pdfplumber'):
            try:
                return self._extract_text_pdfplumber(pdf_path)
            except Exception as e:
                logger.error(f"pdfplumber extraction failed: {e}")

        # Fallback to PyPDF2
        if self.backends_available.get('pypdf2'):
            try:
                return self._extract_text_pypdf2(pdf_path)
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")

        raise RuntimeError("All PDF extraction backends failed")

    def _extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (best quality)"""
        import fitz

        text = []
        doc = fitz.open(pdf_path)

        for page in doc:
            text.append(page.get_text())

        doc.close()

        return "\n\n".join(text)

    def _extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber"""
        import pdfplumber

        text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        return "\n\n".join(text)

    def _extract_text_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2"""
        import PyPDF2

        text = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                text.append(page.extract_text())

        return "\n\n".join(text)

    # ========================================================================
    # Table Extraction Methods
    # ========================================================================

    def _extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        # Try Camelot first (best for tables)
        if self.backends_available.get('camelot'):
            try:
                return self._extract_tables_camelot(pdf_path)
            except Exception as e:
                logger.error(f"Camelot table extraction failed: {e}")

        # Fallback to pdfplumber
        if self.backends_available.get('pdfplumber'):
            try:
                return self._extract_tables_pdfplumber(pdf_path)
            except Exception as e:
                logger.error(f"pdfplumber table extraction failed: {e}")

        return []

    def _extract_tables_camelot(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using Camelot"""
        import camelot

        tables = []

        # Extract tables
        camelot_tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

        for i, table in enumerate(camelot_tables):
            tables.append({
                'table_number': i + 1,
                'page': table.page,
                'data': table.df.to_dict('records'),
                'headers': table.df.columns.tolist(),
                'accuracy': table.accuracy
            })

        return tables

    def _extract_tables_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        import pdfplumber

        tables = []
        table_number = 1

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()

                for table_data in page_tables:
                    if table_data and len(table_data) > 1:
                        # First row as headers
                        headers = table_data[0]
                        rows = table_data[1:]

                        # Convert to dict records
                        records = []
                        for row in rows:
                            record = dict(zip(headers, row))
                            records.append(record)

                        tables.append({
                            'table_number': table_number,
                            'page': page_number,
                            'headers': headers,
                            'data': records
                        })

                        table_number += 1

        return tables

    # ========================================================================
    # Image Extraction Methods
    # ========================================================================

    def _extract_images(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        if not self.backends_available.get('pymupdf'):
            logger.warning("PyMuPDF required for image extraction")
            return []

        import fitz

        images = []
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, 1):
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                images.append({
                    'page': page_number,
                    'image_index': img_index + 1,
                    'format': base_image['ext'],
                    'size': len(base_image['image']),
                    'width': base_image['width'],
                    'height': base_image['height'],
                    'colorspace': base_image.get('colorspace'),
                    'image_data': base_image['image']  # Binary data
                })

        doc.close()

        return images

    # ========================================================================
    # Metadata Extraction
    # ========================================================================

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata"""
        if self.backends_available.get('pymupdf'):
            import fitz
            doc = fitz.open(pdf_path)
            metadata = dict(doc.metadata)
            metadata['page_count'] = len(doc)
            doc.close()
            return metadata

        elif self.backends_available.get('pypdf2'):
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata or {}
                metadata['page_count'] = len(reader.pages)
                return {k.replace('/', ''): v for k, v in metadata.items()}

        return {'page_count': 0}

    # ========================================================================
    # Document Analysis Methods
    # ========================================================================

    def _analyze_document(
        self,
        text: str,
        tables: List[Dict],
        document_type: DocumentType
    ) -> Dict[str, Any]:
        """Analyze document based on type"""
        analysis = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'table_count': len(tables)
        }

        # Add type-specific analysis
        if document_type == DocumentType.EARNINGS_REPORT:
            analysis['contains_earnings_keywords'] = self._check_earnings_keywords(text)
        elif document_type == DocumentType.FINANCIAL_STATEMENT:
            analysis['contains_financial_keywords'] = self._check_financial_keywords(text)

        return analysis

    def _check_earnings_keywords(self, text: str) -> bool:
        """Check if text contains earnings report keywords"""
        keywords = ['revenue', 'earnings per share', 'eps', 'guidance', 'outlook', 'quarterly results']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    def _check_financial_keywords(self, text: str) -> bool:
        """Check if text contains financial statement keywords"""
        keywords = ['assets', 'liabilities', 'equity', 'cash flow', 'operating income']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    # ========================================================================
    # Earnings-Specific Extraction
    # ========================================================================

    def _extract_revenue(self, text: str, tables: List[Dict]) -> Optional[Dict[str, Any]]:
        """Extract revenue from earnings report"""
        # Look for revenue in text
        revenue_pattern = r'revenue.*?\$?([\d,\.]+)\s*(million|billion)?'
        match = re.search(revenue_pattern, text, re.IGNORECASE)

        if match:
            amount = match.group(1).replace(',', '')
            unit = match.group(2) or 'unknown'
            return {'amount': amount, 'unit': unit}

        # Look in tables
        for table in tables:
            for row in table.get('data', []):
                for key, value in row.items():
                    if key and 'revenue' in str(key).lower():
                        return {'amount': value, 'source': 'table'}

        return None

    def _extract_eps(self, text: str, tables: List[Dict]) -> Optional[Dict[str, Any]]:
        """Extract EPS from earnings report"""
        eps_pattern = r'(earnings per share|eps).*?\$?([\d\.]+)'
        match = re.search(eps_pattern, text, re.IGNORECASE)

        if match:
            return {'amount': match.group(2)}

        # Look in tables
        for table in tables:
            for row in table.get('data', []):
                for key, value in row.items():
                    if key and 'eps' in str(key).lower():
                        return {'amount': value, 'source': 'table'}

        return None

    def _extract_guidance(self, text: str) -> Optional[str]:
        """Extract forward guidance from earnings report"""
        guidance_keywords = ['guidance', 'outlook', 'expects', 'forecast']

        for keyword in guidance_keywords:
            # Find paragraph containing guidance
            pattern = rf'({keyword}.*?)(?:\n\n|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)[:500]  # First 500 chars

        return None

    def _extract_key_metrics(self, text: str, tables: List[Dict]) -> List[Dict[str, Any]]:
        """Extract key business metrics"""
        metrics = []

        # Common metric patterns
        metric_patterns = [
            (r'(gross margin).*?([\d\.]+)%', 'gross_margin'),
            (r'(operating margin).*?([\d\.]+)%', 'operating_margin'),
            (r'(net income).*?\$?([\d,\.]+)', 'net_income'),
            (r'(users?|customers?).*?([\d,\.]+)\s*(million|thousand)?', 'user_count'),
        ]

        for pattern, metric_name in metric_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics.append({
                    'metric': metric_name,
                    'value': match.group(2).replace(',', ''),
                    'unit': match.group(3) if len(match.groups()) > 2 else None
                })

        return metrics

    def _extract_management_commentary(self, text: str) -> Optional[str]:
        """Extract CEO/CFO commentary"""
        commentary_pattern = r'(CEO|CFO|management).*?(?:said|stated|commented):(.*?)(?:\n\n|\Z)'
        match = re.search(commentary_pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(2).strip()[:1000]  # First 1000 chars

        return None

    def _extract_quarter(self, text: str) -> Optional[str]:
        """Extract fiscal quarter"""
        quarter_pattern = r'(Q[1-4]|[Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth)\s+quarter'
        match = re.search(quarter_pattern, text)

        if match:
            return match.group(1)

        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """Extract fiscal year"""
        year_pattern = r'(20\d{2})'
        match = re.search(year_pattern, text)

        if match:
            return match.group(1)

        return None

    def _parse_financial_tables(
        self,
        tables: List[Dict],
        statement_type: str
    ) -> List[Dict[str, Any]]:
        """Parse financial statement tables"""
        parsed_data = []

        for table in tables:
            # Look for financial data patterns
            for row in table.get('data', []):
                # Filter rows with numeric values
                numeric_values = {k: v for k, v in row.items() if self._is_numeric(v)}

                if numeric_values:
                    parsed_data.append(row)

        return parsed_data

    def _is_numeric(self, value: str) -> bool:
        """Check if string contains numeric value"""
        if not value:
            return False

        # Remove common formatting
        cleaned = str(value).replace(',', '').replace('$', '').replace('%', '').strip()

        try:
            float(cleaned)
            return True
        except ValueError:
            return False


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_pdf_document(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Quick function to parse any PDF

    Args:
        pdf_path: Path to PDF file

    Returns:
        Parsed document data
    """
    parser = PDFParser()
    return parser.parse_pdf(pdf_path)


def parse_earnings_pdf(
    pdf_path: Union[str, Path],
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to parse earnings report

    Args:
        pdf_path: Path to earnings report PDF
        ticker: Stock ticker

    Returns:
        Earnings data
    """
    parser = PDFParser()
    return parser.parse_earnings_report(pdf_path, ticker)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test PDF parser
    parser = PDFParser()

    print("Available backends:")
    print(json.dumps(parser.backends_available, indent=2))

    # Test with a sample PDF (replace with actual path)
    # result = parser.parse_pdf("path/to/document.pdf")
    # print(json.dumps(result, indent=2))

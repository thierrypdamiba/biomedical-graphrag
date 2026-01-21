import asyncio
import random

from Bio import Entrez

from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class GeneAPIClient:
    """
    Entrez client for the NCBI Gene database (async).
    Fetches gene metadata and linked PubMed papers.
    """

    def __init__(self) -> None:
        # Entrez globals configured by collectors via BaseDataSource
        ...

    async def elink_pubmed_to_gene(self, pmids: list[str]) -> dict[str, list[str]]:
        """
        Map PubMed PMIDs to GeneIDs using Entrez elink (dbfrom=pubmed, db=gene) (async).
        Returns a dict {pmid: [gene_id, ...]}.
        """
        if not pmids:
            return {}

        chunk_size = 50
        max_retries = 3
        base_backoff = 0.8

        pmid_to_genes: dict[str, list[str]] = {}
        for start in range(0, len(pmids), chunk_size):
            chunk = pmids[start : start + chunk_size]
            attempt = 0
            while True:
                try:

                    def _elink(chunk_ids: list[str] = chunk) -> list:
                        handle = Entrez.elink(dbfrom="pubmed", db="gene", id=",".join(chunk_ids))
                        record = Entrez.read(handle)
                        handle.close()
                        return record

                    record = await asyncio.to_thread(_elink)
                    break
                except Exception:
                    attempt += 1
                    if attempt > max_retries:
                        record = []
                        # Fallback to per-ID
                        for pmid in chunk:
                            per_attempt = 0
                            while True:
                                try:

                                    def _single_elink(pmid_id: str = pmid) -> list:
                                        h = Entrez.elink(dbfrom="pubmed", db="gene", id=pmid_id)
                                        single = Entrez.read(h)
                                        h.close()
                                        return single

                                    single = await asyncio.to_thread(_single_elink)
                                    record.extend(single)
                                    break
                                except Exception:
                                    per_attempt += 1
                                    if per_attempt > max_retries:
                                        break
                                    await asyncio.sleep(
                                        base_backoff * (2 ** (per_attempt - 1))
                                        + random.SystemRandom().uniform(0, 0.4)
                                    )
                        break
                    await asyncio.sleep(
                        base_backoff * (2 ** (attempt - 1)) + random.SystemRandom().uniform(0, 0.4)
                    )

            for r in record:
                id_list = r.get("IdList", [])
                pmid = id_list[0] if id_list else ""
                if not pmid:
                    continue
                genes: list[str] = []
                for linkdb in r.get("LinkSetDb", []):
                    if linkdb.get("DbTo") == "gene":
                        genes.extend(
                            [link.get("Id", "") for link in linkdb.get("Link", []) if link.get("Id")]
                        )
                pmid_to_genes[str(pmid)] = sorted(set(genes))
            await asyncio.sleep(0.3)

        return pmid_to_genes

    async def fetch_genes(self, gene_ids: list[str]) -> list[dict]:
        """Fetch structured gene summaries using ESummary (async)."""
        if not gene_ids:
            return []

        def _fetch() -> list[dict]:
            ids = ",".join(gene_ids)
            handle = Entrez.esummary(db="gene", id=ids, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            # Normalize possible wrapper structures
            if isinstance(records, dict) and "DocumentSummarySet" in records:
                summaries = records["DocumentSummarySet"]["DocumentSummary"]
            elif isinstance(records, list):
                summaries = records
            else:
                summaries = [records]

            logger.info(f"Fetched {len(summaries)} gene summaries.")
            return summaries

        return await asyncio.to_thread(_fetch)

    async def link_pubmed(self, gene_id: str) -> list[str]:
        """
        For a given GeneID, return linked PubMed PMIDs via Entrez elink (async).
        """

        def _link() -> list[str]:
            handle = Entrez.elink(dbfrom="gene", db="pubmed", id=gene_id)
            record = Entrez.read(handle)
            handle.close()

            pmids: list[str] = []
            try:
                for linkset in record[0].get("LinkSetDb", []):
                    if linkset.get("DbTo") == "pubmed":
                        pmids.extend([link["Id"] for link in linkset.get("Link", [])])
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to parse PubMed links for GeneID {gene_id}: {exc}")

            logger.info(f"GeneID {gene_id} linked to {len(pmids)} PubMed PMIDs.")
            return pmids

        return await asyncio.to_thread(_link)

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from Bio import Entrez

from biomedical_graphrag.config import settings
from biomedical_graphrag.data_sources.base import BaseDataSource
from biomedical_graphrag.data_sources.gene.gene_api_client import GeneAPIClient
from biomedical_graphrag.domain.dataset import GeneDataset, GeneMetadata, GeneRecord
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class GeneDataCollector(BaseDataSource):
    """
    Collects structured gene information from NCBI Gene,
    and enriches each gene with linked PubMed PMIDs (async).
    """

    def __init__(self) -> None:
        """
        Initialize the GeneDataCollector with a GeneAPIClient instance.

        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.api = GeneAPIClient()

    # Implement abstract method returning the typed entity (dicts for genes here)
    async def fetch_entities(self, entity_ids: list[str]) -> list[Any]:
        """
        Fetch gene entities from the GeneAPIClient.

        Args:
            entity_ids (list[str]): List of GeneIDs to fetch.
        Returns:
            list[dict[str, Any]]: List of gene entities.
        """
        await self._rate_limit()
        return await self.api.fetch_genes(entity_ids)

    async def _batch_link_pubmed(self, gene_ids: list[str]) -> dict[str, list[str]]:
        """
        Batch-fetch PubMed links for multiple GeneIDs in slow, small chunks (async).
        Returns a dict {gene_id: [pmid1, pmid2, ...]}.
        """
        chunk_size = 10
        max_retries = 3
        base_backoff = 0.8

        linked_map: dict[str, list[str]] = {}
        for start in range(0, len(gene_ids), chunk_size):
            chunk = gene_ids[start : start + chunk_size]
            logger.debug(
                f"Linking PubMed for GeneIDs chunk {start}-{start + len(chunk) - 1} (size={len(chunk)})"
            )
            attempt = 0
            while True:
                try:
                    await self._rate_limit()

                    def _elink(chunk_ids: list[str] = chunk, start_idx: int = start) -> list:
                        handle = Entrez.elink(dbfrom="gene", db="pubmed", id=",".join(chunk_ids))
                        record = Entrez.read(handle)
                        handle.close()
                        logger.debug(
                            f"Chunk {start_idx}-{start_idx + len(chunk_ids) - 1} "
                            f"elink succeeded with {len(record)} records"
                        )
                        return record

                    record = await asyncio.to_thread(_elink)
                    break
                except Exception:  # noqa: BLE001
                    attempt += 1
                    if attempt > max_retries:
                        # Fallback to per-ID linking with cautious delays
                        logger.warning(
                            f"Chunk {start}-{start + len(chunk) - 1} failed after retries; \
                            falling back to per-ID linking"
                        )
                        record = []
                        for gid in chunk:
                            per_attempt = 0
                            while True:
                                try:
                                    await self._rate_limit()

                                    def _single_elink(gene_id: str = gid) -> list:
                                        handle = Entrez.elink(dbfrom="gene", db="pubmed", id=gene_id)
                                        single = Entrez.read(handle)
                                        handle.close()
                                        logger.debug(
                                            f"Per-ID elink succeeded for GeneID={gene_id} "
                                            f"with {len(single)} records"
                                        )
                                        return single

                                    single = await asyncio.to_thread(_single_elink)
                                    record.extend(single)
                                    break
                                except Exception:
                                    per_attempt += 1
                                    if per_attempt > max_retries:
                                        logger.warning(
                                            f"Per-ID elink permanently failed for GeneID={gid}"
                                        )
                                        break
                                    jitter = random.SystemRandom().uniform(0, 0.4)
                                    await asyncio.sleep(base_backoff * (2 ** (per_attempt - 1)) + jitter)
                        break
                    jitter = random.SystemRandom().uniform(0, 0.4)
                    await asyncio.sleep(base_backoff * (2 ** (attempt - 1)) + jitter)

            for linkset in record:
                gene_id = linkset.get("IdList", [None])[0]
                pmids: list[str] = []
                for linkdb in linkset.get("LinkSetDb", []):
                    if linkdb.get("DbTo") == "pubmed":
                        pmids.extend([link["Id"] for link in linkdb.get("Link", [])])
                if gene_id:
                    linked_map[gene_id] = pmids

            # gentle pause between chunks
            await asyncio.sleep(0.35)

        logger.info(f"Linked PubMed PMIDs for {len(linked_map)} genes.")
        return linked_map

    async def collect_dataset(self, query: str = "", max_results: int = 0) -> GeneDataset:
        """
        Collect gene metadata by resolving GeneIDs from PubMed PMIDs (async).

        Returns:
            GeneDataset: The collected gene dataset.
        """
        logger.info("Collecting Gene dataset from PubMed PMIDs...")
        # Load PMIDs from the existing PubMed dataset
        pubmed_path = settings.json_data.pubmed_json_path
        try:

            def _load() -> dict:
                with open(pubmed_path, encoding="utf-8") as f:
                    return json.load(f)

            pubmed_ds: dict = await asyncio.to_thread(_load)
        except FileNotFoundError:
            logger.error(f"PubMed dataset not found at {pubmed_path}")
            return GeneDataset()
        pmids = [str(p.get("pmid", "")) for p in pubmed_ds.get("papers", []) if str(p.get("pmid", ""))]
        if not pmids:
            logger.warning("No PMIDs found in PubMed dataset.")
            return GeneDataset()

        # Fetch metadata
        # Map PMIDs -> GeneIDs using elink
        logger.info(f"Resolving GeneIDs from {len(pmids)} PMIDs via elink")
        pmid_to_genes = await self.api.elink_pubmed_to_gene(pmids)
        all_gene_ids = sorted({gid for genes in pmid_to_genes.values() for gid in genes})
        logger.info(f"Resolved {len(all_gene_ids)} unique GeneIDs; fetching summaries")
        await self._rate_limit()
        gene_summaries = await self.api.fetch_genes(all_gene_ids)
        logger.info(f"Fetched {len(gene_summaries)} gene summaries; computing linked PMIDs per gene")

        # Batch link to PubMed (1 elink call for all GeneIDs)
        logger.info("Fetching PubMed links for all GeneIDs...")
        # Invert mapping to gene_id -> linked_pmids
        linked_map: dict[str, list[str]] = {}
        for pmid, genes in pmid_to_genes.items():
            for gid in genes:
                linked_map.setdefault(gid, []).append(pmid)

        gene_records: list[GeneRecord] = []
        for gene in gene_summaries:
            gene_id = gene.get("uid", "")
            linked_pmids = linked_map.get(gene_id, [])

            gene_records.append(
                GeneRecord(
                    gene_id=gene_id,
                    name=gene.get("Name", ""),
                    description=gene.get("Description", gene.get("Summary", "")),
                    chromosome=gene.get("Chromosome", ""),
                    map_location=gene.get("MapLocation", ""),
                    organism=gene.get("Organism", {}).get("ScientificName", ""),
                    aliases=gene.get("OtherAliases", ""),
                    designations=gene.get("OtherDesignations", ""),
                    linked_pmids=linked_pmids,
                )
            )

        total_linked = sum(len(r.linked_pmids) for r in gene_records)
        with_links = sum(1 for r in gene_records if r.linked_pmids)
        metadata = GeneMetadata(
            collection_date=datetime.now().isoformat(),
            total_genes=len(gene_records),
            genes_with_pubmed_links=with_links,
            total_linked_pmids=total_linked,
        )
        logger.info(
            f"✅ Collected {len(gene_records)} gene entries "
            f"(with_pubmed_links={with_links}, total_linked_pmids={total_linked})."
        )
        return GeneDataset(metadata=metadata, genes=gene_records)


if __name__ == "__main__":

    async def main() -> None:
        """Main function to collect gene dataset and save to file."""
        collector = GeneDataCollector()
        gene_ds = await collector.collect_dataset()

        Path("data").mkdir(exist_ok=True)

        def _save() -> None:
            with open(settings.json_data.gene_json_path, "w") as f:
                f.write(gene_ds.model_dump_json(indent=2))

        await asyncio.to_thread(_save)
        print(f"✅ Saved {len(gene_ds.genes)} genes to data/gene_dataset.json")

    asyncio.run(main())

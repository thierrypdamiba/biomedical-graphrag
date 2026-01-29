import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

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

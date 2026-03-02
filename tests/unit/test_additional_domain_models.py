"""Unit tests for additional domain models."""

from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.dataset import (
    GeneDataset,
    GeneMetadata,
    PaperDataset,
    PaperMetadata,
)
from biomedical_graphrag.domain.gene import GeneRecord


class TestGeneRecord:
    def test_default_creation(self) -> None:
        gene = GeneRecord()
        assert gene.gene_id == ""
        assert gene.name == ""
        assert gene.linked_pmids == []

    def test_full_creation(self) -> None:
        gene = GeneRecord(
            gene_id="672",
            name="BRCA1",
            description="BRCA1 DNA repair associated",
            chromosome="17",
            map_location="17q21.31",
            organism="Homo sapiens",
            aliases="BRCA1/BRCA2-containing complex",
            linked_pmids=["12345", "67890"],
        )
        assert gene.gene_id == "672"
        assert gene.name == "BRCA1"
        assert len(gene.linked_pmids) == 2


class TestCitationNetwork:
    def test_default_creation(self) -> None:
        citation = CitationNetwork()
        assert citation.pmid == ""
        assert citation.cited_by == []
        assert citation.references == []

    def test_with_data(self) -> None:
        citation = CitationNetwork(
            pmid="12345",
            cited_by=["11111", "22222"],
            references=["33333"],
        )
        assert len(citation.cited_by) == 2
        assert len(citation.references) == 1


class TestDatasetModels:
    def test_paper_metadata_defaults(self) -> None:
        meta = PaperMetadata()
        assert meta.total_papers == 0
        assert meta.collection_date == ""

    def test_paper_dataset_defaults(self) -> None:
        ds = PaperDataset()
        assert ds.papers == []
        assert ds.citation_network == {}

    def test_gene_metadata_defaults(self) -> None:
        meta = GeneMetadata()
        assert meta.total_genes == 0
        assert meta.genes_with_pubmed_links == 0

    def test_gene_dataset_defaults(self) -> None:
        ds = GeneDataset()
        assert ds.genes == []

    def test_gene_dataset_with_data(self) -> None:
        gene = GeneRecord(gene_id="672", name="BRCA1")
        ds = GeneDataset(
            metadata=GeneMetadata(total_genes=1, genes_with_pubmed_links=1),
            genes=[gene],
        )
        assert ds.metadata.total_genes == 1
        assert len(ds.genes) == 1
        assert ds.genes[0].name == "BRCA1"

"""
Majority Judgement Ballot Generator

This module provides functionality to generate majority judgement ballots
in various paper sizes. It uses matplotlib to create the ballots and save
them as PDF files.

Classes:
    PaperSize: Enum for supported paper sizes.
    BallotConfig: Configuration for ballot generation.
    BallotGenerator: Main class for generating ballots.

Constants:
    PAPER_DIMENSIONS: Dictionary of paper dimensions.
    PAPER_FONT_SCALE: Dictionary of font scaling factors.
    PAPER_ZOOM_FACTOR: Dictionary of zoom factors for images.
    DEFAULT_GRADES: List of default grades.
    SMALL_SENTENCE: Informational text for the ballot.
    PATH_LOGO_MIEUX_VOTER: Path to the Mieux Voter logo.
    PATH_QR_CODE: Path to the QR code image.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import font_manager
from PIL import Image
import numpy as np
import random
from enum import Enum
from typing import List, Dict, Tuple
import itertools

# Constants
PAPER_DIMENSIONS: Dict[str, Tuple[float, float]] = {"A4": (11.69, 8.27), "A5": (8.27, 5.83), "A6": (5.83, 4.13)}
PAPER_FONT_SCALE: Dict[str, float] = {"A4": 1, "A5": 0.7, "A6": 0.5}
PAPER_ZOOM_FACTOR: Dict[str, float] = {"A4": 1, "A5": 0.5, "A6": 0.25}
DEFAULT_GRADES: List[str] = ["To reject", "Insufficient", "Passable", "Fair", "Good", "Very good", "Excellent"]
SMALL_SENTENCE: str = (
    "Ce bulletin est produit par Mieux Voter pour une élection plus juste. \n"
    "Veuillez visiter mieuxvoter.fr pour plus d'informations."
)
PATH_LOGO_MIEUX_VOTER: str = "doc/logo.png"
PATH_QR_CODE: str = "doc/qrcode_mieux_voter.png"
BASE_COLORS: List[str] = ["#990000", "#C23D13", "#C27C13", "#C2B113", "#D3D715", "#A0CF1C", "#3A9918"]


# Set up font
font_manager.fontManager.addfont("font/Lato-Black.ttf")
plt.rcParams["font.family"] = "Lato-Black"


class PaperSize(Enum):
    """Enum for supported paper sizes."""

    A4 = "A4"
    A5 = "A5"
    A6 = "A6"


class BallotConfig:
    """
    Configuration class for ballot generation.

    Attributes:
        question (str): The main question on the ballot.
        candidates (List[str]): List of candidate names.
        grades (List[str]): List of grade options.
    """

    def __init__(self, question: str, candidates: List[str], grades: List[str] = None):
        """
        Initialize BallotConfig.

        Parameters:
            question (str): The main question on the ballot.
            candidates (List[str]): List of candidate names.
            grades (List[str], optional): List of grade options. Defaults to DEFAULT_GRADES.
        """
        self.question = question
        self.candidates = candidates
        self.grades = grades or DEFAULT_GRADES


class BallotGenerator:
    """
    Main class for generating majority judgement ballots.

    This class handles the creation and saving of ballots based on
    the provided configuration.
    """

    def __init__(self, config: BallotConfig):
        """
        Initialize BallotGenerator.

        Parameters:
            config (BallotConfig): Configuration for ballot generation.
        """
        self.config = config
        self.grade_colors = self._generate_grade_colors()

    def generate_ballot(self, output_filename: str, paper_size: PaperSize = PaperSize.A4):
        """
        Generate and save a ballot as a PDF file.

        Parameters:
            output_filename (str): Name of the output PDF file.
            paper_size (PaperSize): Size of the paper to use.

        Raises:
            ValueError: If an unsupported paper size is provided.
        """
        if paper_size not in PaperSize:
            raise ValueError(f"Unsupported paper size: {paper_size}")

        fig, ax = self._create_figure(paper_size)
        self._add_ballot_content(ax, paper_size)
        self._save_ballot(fig, output_filename)

    def generate_shuffled_ballots(
        self,
        nb_ballots: int,
        output_filename: str = None,
        paper_size: PaperSize = PaperSize.A4,
    ):
        """
        Generate and save multiple ballots with mixed candidate orders as a single PDF file.

        This method efficiently generates ballots by pre-generating all unique combinations
        when the number of combinations is less than the requested number of ballots.
        It then adds these pre-generated figures to the PDF in random order.

        Parameters:
            nb_ballots (int): Number of ballots to generate.
            output_filename (str): Name of the output PDF file.
            paper_size (PaperSize): Size of the paper to use.

        Raises:
            ValueError: If an unsupported paper size is provided.
        """
        if paper_size not in PaperSize:
            raise ValueError(f"Unsupported paper size: {paper_size}")

        output_filename = (
            output_filename or f"mixed_{nb_ballots}_majority_judgement_ballots_{paper_size.value}_landscape.pdf"
        )

        # Generate all possible permutations of candidates
        all_permutations = list(itertools.permutations(self.config.candidates))
        total_permutations = len(all_permutations)

        # Determine whether to use all permutations or random shuffling
        use_all_permutations = total_permutations <= nb_ballots

        # Pre-generate all figures
        figures = []
        if use_all_permutations:
            for candidates in all_permutations:
                fig, ax = self._create_figure(paper_size)
                self._add_ballot_content(ax, paper_size, candidates=candidates)
                figures.append(fig)

            # If we need more ballots than permutations, duplicate some figures
            while len(figures) < nb_ballots:
                figures.append(random.choice(figures))
        else:
            np.random.seed(42)
            unique_indices = np.random.choice(total_permutations, nb_ballots, replace=False)

            for idx in unique_indices:
                fig, ax = self._create_figure(paper_size)

                candidates = all_permutations[idx]
                self._add_ballot_content(ax, paper_size, candidates=candidates)
                figures.append(fig)

        # Randomly shuffle the figures
        random.shuffle(figures)

        # Save the figures to PDF
        with PdfPages(output_filename) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches="tight", orientation="landscape", dpi=600)
                plt.close(fig)

        print(f"{nb_ballots} mixed ballots saved as '{output_filename}'")

    def _create_figure(self, paper_size: PaperSize) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a matplotlib figure and axes for the ballot.

        Parameters:
            paper_size (PaperSize): Size of the paper to use.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The created figure and axes.
        """
        paper_width, paper_height = PAPER_DIMENSIONS[paper_size.value]
        fig_width, fig_height = paper_width - 1, paper_height - 1
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        return fig, ax

    def _add_ballot_content(self, ax: plt.Axes, paper_size: PaperSize, candidates: List[str] = None):
        """
        Add all content to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            paper_size (PaperSize): Size of the paper being used.
            shuffle_candidates (bool): Whether to shuffle the order of candidates.
        """
        font_size = self._calculate_font_size(paper_size)
        self._add_question(ax, font_size)
        self._add_logo(ax, paper_size)
        self._add_qr_code(ax, paper_size)
        self._add_small_sentence(ax, font_size)
        self._add_table(ax, font_size, candidates)

    def _calculate_font_size(self, paper_size: PaperSize) -> float:
        """
        Calculate the appropriate font size based on paper size and content.

        Parameters:
            paper_size (PaperSize): Size of the paper being used.

        Returns:
            float: The calculated font size.
        """
        return min(
            9, 60 / max(len(self.config.candidates), len(self.config.grades)) * PAPER_FONT_SCALE[paper_size.value]
        )

    def _add_question(self, ax: plt.Axes, font_size: float):
        """
        Add the main question to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            font_size (float): The base font size to use.
        """
        ax.text(0.5, 0.97, self.config.question, fontsize=font_size * 1.5, fontweight="bold", ha="center", va="top")

    def _add_logo(self, ax: plt.Axes, paper_size: PaperSize):
        """
        Add the Mieux Voter logo to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            paper_size (PaperSize): Size of the paper being used.
        """
        logo = plt.imread(PATH_LOGO_MIEUX_VOTER)
        imagebox = OffsetImage(logo, zoom=0.35 * PAPER_ZOOM_FACTOR[paper_size.value])
        ab = AnnotationBbox(imagebox, (0.95, 0.015), frameon=False, xycoords="axes fraction", box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    def _add_qr_code(self, ax: plt.Axes, paper_size: PaperSize):
        """
        Add the QR code to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            paper_size (PaperSize): Size of the paper being used.
        """
        qr_code = Image.open(PATH_QR_CODE).convert("RGB")
        qr_code = np.array(qr_code)
        imagebox = OffsetImage(qr_code, zoom=0.1 * PAPER_ZOOM_FACTOR[paper_size.value])
        ab = AnnotationBbox(imagebox, (0.05, 0.01), frameon=False, xycoords="axes fraction", box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    def _add_small_sentence(self, ax: plt.Axes, font_size: float):
        """
        Add the small informational sentence to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            font_size (float): The base font size to use.
        """
        ax.text(0.5, 0.02, SMALL_SENTENCE, fontsize=font_size * 0.75, ha="center", va="top")

    def _add_table(self, ax: plt.Axes, font_size: float, candidates: List[str] = None):
        """
        Add the main table to the ballot.

        Parameters:
            ax (plt.Axes): The axes to draw on.
            font_size (float): The font size to use for the table.
            candidates (List[str], optional): List of candidate names. Defaults to None.
        """
        candidates = self.config.candidates.copy() if candidates is None else candidates

        data = [["" for _ in self.config.grades] for _ in candidates]
        table = ax.table(
            cellText=data,
            rowLabels=candidates,
            colLabels=self.config.grades,
            cellLoc="center",
            loc="center",
            bbox=[0.05, 0.05, 0.95, 0.85],
        )
        self._format_table(table, font_size)

    def _format_table(self, table: plt.table, font_size: float):
        """
        Format the table on the ballot.

        Parameters:
            table (plt.Table): The table to format.
            font_size (float): The font size to use.
        """
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.5)

        for j, grade in enumerate(self.config.grades):
            table[(0, j)].set_facecolor(self.grade_colors[j])
            table[(0, j)].set_text_props(color="white", weight="bold")

        for cell in table._cells:
            table._cells[cell].set_edgecolor("black")
            table._cells[cell].set_linewidth(0.5)

    def _save_ballot(self, fig: plt.Figure, output_filename: str):
        """
        Save the ballot as a PDF file.

        Parameters:
            fig (plt.Figure): The figure to save.
            output_filename (str): Name of the output PDF file.
        """
        plt.tight_layout()
        with PdfPages(output_filename) as pdf:
            pdf.savefig(fig, bbox_inches="tight", orientation="landscape", dpi=600)
        plt.close(fig)

    def _generate_grade_colors(self) -> List[str]:
        """
        Generate a list of colors for the grades.

        Returns:
            List[str]: List of color codes for each grade.
        """
        num_grades = len(self.config.grades)

        if num_grades > len(BASE_COLORS):
            raise ValueError("More grades than available colors")

        extra_colors = len(BASE_COLORS) - num_grades
        start_index = extra_colors // 2
        return BASE_COLORS[start_index : start_index + num_grades]


def main():
    """
    Main function to demonstrate the usage of the BallotGenerator.
    """
    question = "Is the candidate suitable to be a baker?"
    candidates = ["Alice", "Bob", "Charlie", "David", "Eve", "Gérard"]
    grades = ["To reject", "Insufficient", "Passable", "Fair", "Good", "Very good", "Excellent"]

    config = BallotConfig(question, candidates, grades)
    generator = BallotGenerator(config)

    for size in PaperSize:
        output_filename = f"majority_judgement_ballot_{size.value}_landscape.pdf"
        generator.generate_ballot(output_filename, size)
        print(f"Ballot saved as '{output_filename}'")

    # Generate 100 mixed ballots with shuffled candidate orders
    # factorial(6) = 720, so we can generate the figures on the fly
    generator.generate_shuffled_ballots(100, paper_size=PaperSize.A4)

    candidates = ["Alice", "Gérard", "Bob"]
    config = BallotConfig(question, candidates, grades)
    generator = BallotGenerator(config)
    # factorial(3) = 6, so we can generate the figures first and shuffle them
    generator.generate_shuffled_ballots(50, paper_size=PaperSize.A4)


if __name__ == "__main__":
    main()

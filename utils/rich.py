import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import pandas as pd
from rich import box, print
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    Text,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.table import Column, Table

console = Console()


class IterationSpeedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task) -> Text:
        """Show time remaining."""

        if not (task.elapsed and task.completed):
            return Text("-- it/s", style="progress.remaining")

        iters = task.completed / task.elapsed
        string = f"{iters:.2f} it/s" if iters > 1 else f"{1/iters:.2f} s/it"
        return Text(string, style="progress.remaining")


@contextmanager
def custom_progress(width=None, verbose=True) -> Iterator[Progress]:
    with Progress(
        TextColumn(
            "[bold blue]{task.description}",
            justify="left",
            table_column=Column(width=width),
        ),
        TextColumn(
            "{task.completed:.0f}/{task.total:.0f}",
            justify="right",
            style="progress.percentage",
        ),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(compact=True),
        TimeElapsedColumn(),
        IterationSpeedColumn(),
        disable=not verbose,
    ) as progress:
        yield progress


simple_progress = [
    TextColumn("[bold blue]{task.description}", justify="left"),
    TextColumn(
        "{task.completed}/{task.total}", justify="right", style="progress.percentage"
    ),
    BarColumn(),
    TaskProgressColumn(show_speed=True),
    TimeRemainingColumn(compact=True),
    TimeElapsedColumn(),
]


def override_file_prompt(filename: Union[str, Path]):
    """

    Args:
        filename (Union[str, Path]): _description_

    Returns:
        bool: True if path can be used, False otherwise
    """

    filename = Path(filename)
    if not filename.exists():
        return True

    if filename.is_file():  # filename exists
        msg = f"file [bold blue]{filename}[/bold blue] exists. [red bold]Override?[/bold red]"
        choice = Prompt.ask(msg, choices=["yes", "no"])
        return choice != "no"

    if filename.is_dir():  # filename is a directory
        print(f"file {filename} is a directory")

    return False


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Optional[Table] = None,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = rich_table or Table(show_header=True, header_style="bold magenta")

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name, justify="center")
        rich_indexes = pandas_dataframe.index.to_list()

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column), justify="center")

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(rich_indexes[index])] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


def dataframes_inline_table(
    dict_dfs: Dict[str, pd.DataFrame],
    table: Optional[Table] = None,
    title: Optional[str] = None,
):
    if table is None:
        table = Table(header_style="bold green", title=title)
        for colname in dict_dfs:
            table.add_column(colname, justify="center")

    table.add_row(*map(df_to_table, dict_dfs.values()))
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    return table


def example_df_to_table():
    sample_data = {
        "Date": [
            datetime(year=2019, month=12, day=20),
            datetime(year=2018, month=5, day=25),
            datetime(year=2017, month=12, day=15),
        ],
        "Title": [
            "Star Wars: The Rise of Skywalker",
            "[red]Solo[/red]: A Star Wars Story",
            "Star Wars Ep. VIII: The Last Jedi",
        ],
        "Production Budget": ["$275,000,000", "$275,000,000", "$262,000,000"],
        "Box Office": ["$375,126,118", "$393,151,347", "$1,332,539,889"],
    }
    df = pd.DataFrame(sample_data)

    # Initiate a Table instance to be modified
    table = Table(show_header=True, header_style="bold magenta")

    # Modify the table instance to have the data from the DataFrame
    table = df_to_table(df, table)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    console.print(table)


def example_prompt():
    choice = override_file_prompt(__file__)
    print(f"Overiding decision is {choice}")

    with custom_progress(width=40) as progress:
        for _ in progress.track(range(100)):
            time.sleep(0.1)


if __name__ == "__main__":
    example_df_to_table()
    example_prompt()

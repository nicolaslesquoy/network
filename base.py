from __future__ import annotations
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Constants
MODE = "DEBUG"  # "PROD"
CAPACITY = 100 * 10**6  # 100 Mbps
PATH_TO_OUT = Path("out/")
assert PATH_TO_OUT.exists()

# Utility functions


def bytes_to_bits(byte: float) -> float:
    return byte * 8


def ms_to_s(milliseconds: float) -> float:
    return milliseconds / 1000


def s_to_ms(seconds: float) -> float:
    return seconds * 10**6


# Classes
class Flow:
    def __init__(
        self,
        name: str,
        overhead: float,
        payload: float,
        period: float,
        source: "Station",
    ) -> None:
        self.name = name
        self.overhead = overhead
        self.payload = payload
        self.period = period
        self.source = source
        self.targets: list["Target"] = []

    def get_data_length(self) -> float:
        return self.overhead + self.payload

    def get_rate(self) -> float:
        return self.get_data_length() / self.period

    def __repr__(self) -> str:
        return f"Flow {self.name}"

    def __str__(self) -> str:
        return f"Flow {self.name}"


class Target:
    def __init__(self, flow: "Flow", destination: "Station") -> None:
        self.path: list["Edge"] = []
        self.flow = flow
        self.destination = destination
        self.current_step = 0
        self.arrivalCurve = ArrivalCurve(flow.get_data_length(), flow.get_rate())
        self.total_delay = 0.0
        self.completed = False


class ArrivalCurve:
    def __init__(self, burst: float, rate: float) -> None:
        self.burst = burst
        self.rate = rate

    def add(self, burst: float, rate: float) -> None:
        self.burst += burst
        self.rate += rate

    def add_delay(self, delay: float) -> None:
        self.burst += delay * self.rate


# Link
class Edge:
    def __init__(self, source: "Node", destination: "Node", name: str) -> None:
        self.source = source
        self.destination = destination
        self.name = name
        self.goal = 0.0  # Calculated at the end of parseXML
        self.arrival_curve_aggregated = ArrivalCurve(
            0, 0
        )  # Will aggregate all curves passing through this Edge
        self.flows_passed: list[str] = []  # Track flows already aggregated
        self.delay = 0.0  # Calculated after all aggregations
        self.load = 0.0  # Calculated at the end, to compare with C

    def __repr__(self) -> str:
        return f"Edge from {self.source.name} to {self.destination.name}"

    def __str__(self) -> str:
        return f"Edge from {self.source.name} to {self.destination.name}"


class Node:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Node {self.name}"

    def __str__(self) -> str:
        return f"Node {self.name}"


class Switch(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)


class Station(Node):
    def __init__(self, name):
        self.arrival_curve_aggregated = ArrivalCurve(0, 0)
        super().__init__(name)


class Parser:
    def __init__(self, file: Path) -> None:
        self.file = file
        self.tree = ET.parse(str(file))
        self.root = self.tree.getroot()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.flows: Dict[str, Flow] = {}
        self.targets: List[Target] = []

    def find_edge(self, source: Node, dest: Node) -> int:
        found = [
            index
            for index in range(len(self.edges))
            if (source.name == self.edges[index].source.name)
            and (dest.name == self.edges[index].destination.name)
        ]
        assert len(found) == 1
        return found[0]

    def parse_stations(self) -> None:
        for station in self.root.findall("station"):
            name = station.get("name")
            if name:
                self.nodes[name] = Station(name)

    def parse_switches(self) -> None:
        for sw in self.root.findall("switch"):
            name = sw.get("name")
            if name:
                self.nodes[name] = Switch(name)

    def parse_edges(self) -> None:
        for link in self.root.findall("link"):
            source_name = link.get("from")
            dest_name = link.get("to")
            name = link.get("name")

            if source_name and dest_name and name:
                source = self.nodes[source_name]
                dest = self.nodes[dest_name]

                edge = Edge(source, dest, name)
                self.edges.append(edge)
                # Create reverse edge
                edge_reverse = Edge(dest, source, name)
                self.edges.append(edge_reverse)

    def parse_flows(self) -> None:
        for fl in self.root.findall("flow"):
            name = fl.get("name")
            source_name = fl.get("source")

            if name and source_name:
                source = self.nodes[source_name]
                max_payload = fl.get("max-payload")
                period = fl.get("period")

                if max_payload and period:
                    flow = Flow(
                        name,
                        bytes_to_bits(67),
                        bytes_to_bits(float(max_payload)),
                        ms_to_s(float(period)),
                        source,
                    )
                    self.flows[name] = flow

                    edges_traversed: List[Edge] = []

                    for tg in fl.findall("target"):
                        dest_name = tg.get("name")
                        if dest_name:
                            dest = self.nodes[dest_name]
                            target = Target(flow, dest)
                            flow.targets.append(target)

                            step_source = source
                            for pt in tg.findall("path"):
                                step_dest_name = pt.get("node")
                                if step_dest_name:
                                    step_dest = self.nodes[step_dest_name]
                                    edge = self.edges[
                                        self.find_edge(step_source, step_dest)
                                    ]
                                    target.path.append(edge)
                                    if edge not in edges_traversed:
                                        edges_traversed.append(edge)
                                    step_source = step_dest

                            self.targets.append(target)

                    for edge in edges_traversed:
                        edge.goal += 1

    def parse_network(self) -> Tuple[Dict[str, Flow], List[Target], List[Edge]]:
        """Parse the network configuration from the XML file."""
        self.parse_stations()
        self.parse_switches()
        self.parse_edges()
        self.parse_flows()
        return self.flows, self.targets, self.edges


class Writer:
    """Handles writing network simulation results to XML files."""

    def __init__(
        self, xml_file: str, mode: str = MODE, path_to_out: Path = PATH_TO_OUT
    ):
        """Initialize the writer with output configuration.

        Args:
            xml_file: Path to the input XML file
            mode: Simulation mode ("DEBUG" or "PROD")
            path_to_out: Directory path for output files
        """
        self.input_path = Path(xml_file)
        self.mode = mode
        self.output_dir = path_to_out
        self._setup_output_path()

    def _setup_output_path(self) -> None:
        """Set up the output file path based on input file and mode."""
        suffix = "_DEBUG" if self.mode == "DEBUG" else ""
        self.output_path = (
            self.output_dir
            / f"{self.input_path.stem}_res{suffix}{self.input_path.suffix}"
        )

    def write_results(self, flows: List[Flow], edges: List[Edge]) -> None:
        """Write network simulation results to XML file.

        Args:
            flows: List of Flow objects with simulation results
            edges: List of Edge objects with load information
        """
        with self.output_path.open("w") as res:
            self._write_header(res)
            self._write_delays(res, flows)
            self._write_loads(res, edges)
            self._write_footer(res)

    def _write_header(self, file) -> None:
        """Write XML header and opening tag."""
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write("<results>\n")

    def _write_delays(self, file, flows) -> None:
        """Write delay information for all flows."""
        file.write("\t<delays>\n")
        for flow in flows:
            file.write(f'\t\t<flow name="{flow.name}">\n')
            for target in flow.targets:
                file.write(
                    f'\t\t\t<target name="{target.destination.name}" '
                    f'value="{s_to_ms(target.total_delay)}" />\n'
                )
            file.write("\t\t</flow>\n")
        file.write("\t</delays>\n")

    def _write_loads(self, file, edges) -> None:
        """Write load information for all edges."""
        file.write("\t<load>\n")
        for i in range(0, len(edges) - 1, 2):
            edge_direct = edges[i]
            edge_reverse = edges[i + 1]
            file.write(f'\t\t<edge name="{edge_direct.name}">\n')
            file.write(
                f'\t\t\t<usage percent="{edge_direct.load:.1f}%" type="direct" '
                f'value="{edge_direct.arrival_curve_aggregated.rate}" />\n'
            )
            file.write(
                f'\t\t\t<usage percent="{edge_reverse.load:.1f}%" type="reverse" '
                f'value="{edge_reverse.arrival_curve_aggregated.rate}" />\n'
            )
            file.write("\t\t</edge>\n")
        file.write("\t</load>\n")

    def _write_footer(self, file) -> None:
        """Write closing tag."""
        file.write("</results>\n")

    def print_output(self) -> None:
        """Print the contents of the output file to console."""
        with self.output_path.open("r") as f:
            for line in f:
                print(line.rstrip())


class NetworkCalculus:
    """Handles network calculus computations for delays and loads."""

    def __init__(
        self, flows: Dict[str, Flow], targets: List[Target], edges: List[Edge]
    ):
        """Initialize with network components.

        Args:
            flows: Dictionary of Flow objects
            targets: List of Target objects
            edges: List of Edge objects
        """
        self.flows = flows
        self.targets = targets
        self.edges = edges
        self.capacity = CAPACITY

    def _add_curve_to_switch(self, target: Target, edge: Edge) -> None:
        """Add arrival curve to edge if not already accounted for."""
        already_accounted = target.flow.name in edge.flows_passed
        if not already_accounted:
            edge.arrival_curve_aggregated.add(
                target.arrivalCurve.burst, target.arrivalCurve.rate
            )
            edge.flows_passed.append(target.flow.name)

    def _check_proceed(self, target: Target) -> bool:
        """Check if target can proceed to next step."""
        if target.completed:
            return False
        edge = target.path[target.current_step]
        return len(edge.flows_passed) == edge.goal

    def calculate_station_arrival_curves(self) -> None:
        """Calculate initial arrival curves for each source station."""
        for flow in self.flows.values():
            flow.source.arrival_curve_aggregated.add(
                flow.get_data_length(), flow.get_rate()
            )

    def process_source_stations(self) -> None:
        """Process initial delays from source stations."""
        for target in self.targets:
            source = target.flow.source
            edge = target.path[target.current_step]

            # Add source station's curve to first edge
            self._add_curve_to_switch(target, edge)

            # Calculate source delay if needed
            if edge.delay == 0:
                edge.delay = source.arrival_curve_aggregated.burst / self.capacity

            # Update arrival curve and move to next step
            target.arrivalCurve.add_delay(edge.delay)
            target.current_step += 1

            # Add curve to next switch
            next_edge = target.path[target.current_step]
            self._add_curve_to_switch(target, next_edge)

    def process_target(self, target: Target) -> bool:
        """Process a single target through its path.

        Returns:
            bool: True if target completed its path, False otherwise
        """
        while self._check_proceed(target):
            edge = target.path[target.current_step]

            # Calculate edge delay if needed
            if edge.delay == 0:
                edge.delay = edge.arrival_curve_aggregated.burst / self.capacity

            target.arrivalCurve.add_delay(edge.delay)

            # Check if target reached destination
            if target.path[target.current_step] == target.path[-1]:
                target.total_delay = sum(edge.delay for edge in target.path)
                target.completed = True
                return True

            # Move to next edge
            target.current_step += 1
            next_edge = target.path[target.current_step]
            self._add_curve_to_switch(target, next_edge)

        return False

    def process_all_targets(self) -> None:
        """Process all targets until completion."""
        targets_completed = 0
        current_index = 0

        while targets_completed < len(self.targets):
            target = self.targets[current_index]
            if not target.completed and self.process_target(target):
                targets_completed += 1
            current_index = (current_index + 1) % len(self.targets)

    def calculate_edge_loads(self) -> None:
        """Calculate final load percentages for all edges."""
        for i in range(0, len(self.edges), 2):
            # Calculate direct traffic load
            direct_edge = self.edges[i]
            direct_edge.load = (
                direct_edge.arrival_curve_aggregated.rate * 100 / self.capacity
            )

            # Calculate reverse traffic load
            reverse_edge = self.edges[i + 1]
            reverse_edge.load = (
                reverse_edge.arrival_curve_aggregated.rate * 100 / self.capacity
            )

            # Ensure loads don't exceed link capacity
            total_load = direct_edge.load + reverse_edge.load
            if total_load > 100:
                print(
                    f"Warning: Total load {total_load:.1f}% exceeds link capacity for {direct_edge.name}"
                )

    def compute(self) -> None:
        """Perform all network calculus computations."""
        self.calculate_station_arrival_curves()
        self.process_source_stations()
        self.process_all_targets()
        self.calculate_edge_loads()


class Check:
    """Handles comparison of simulation results against reference data."""

    def __init__(
        self, input_file: str, mode: str = MODE, path_to_out: Path = PATH_TO_OUT
    ):
        self.sim_path = (
            path_to_out
            / f"{Path(input_file).stem}_res{'_DEBUG' if mode=='DEBUG' else ''}.xml"
        )
        self.ref_path = Path(input_file).parent / f"{Path(input_file).stem}_res.xml"

    def _compare_delays(
        self, sim_tree: ET.ElementTree, ref_tree: ET.ElementTree
    ) -> tuple[float, float]:
        """Compare delays between simulation and reference results."""
        print("\n=== Delay Comparison ===")
        print("------------------------")

        sim_delays = sim_tree.findall(".//flow")
        ref_delays = ref_tree.findall(".//flow")

        max_diff = 0.0
        total_diff = 0.0
        count = 0

        for sim_flow, ref_flow in zip(sim_delays, ref_delays):
            flow_name = sim_flow.get("name")
            print(f"\nFlow: {flow_name}")
            print("-" * (len(flow_name) + 7))

            sim_targets = sim_flow.findall("target")
            ref_targets = ref_flow.findall("target")

            for sim_target, ref_target in zip(sim_targets, ref_targets):
                target_name = sim_target.get("name")
                sim_delay = float(sim_target.get("value"))
                ref_delay = float(ref_target.get("value"))
                diff = abs(sim_delay - ref_delay)

                status = "✓" if diff < 1.0 else "!"
                print(
                    f"{status} Target {target_name:10} - Sim: {sim_delay:8.1f}μs, "
                    f"Ref: {ref_delay:8.1f}μs, Diff: {diff:8.1f}μs"
                )

                max_diff = max(max_diff, diff)
                total_diff += diff
                count += 1

        return max_diff, (total_diff / count if count > 0 else 0)

    def _compare_loads(
        self, sim_tree: ET.ElementTree, ref_tree: ET.ElementTree
    ) -> tuple[float, float]:
        """Compare loads between simulation and reference results."""
        print("\n=== Load Comparison ===")
        print("----------------------")

        sim_edges = sim_tree.findall(".//edge")
        ref_edges = ref_tree.findall(".//edge")

        max_diff = 0.0
        total_diff = 0.0
        count = 0

        for sim_edge, ref_edge in zip(sim_edges, ref_edges):
            edge_name = sim_edge.get("name")
            print(f"\nEdge: {edge_name}")
            print("-" * (len(edge_name) + 7))

            sim_usages = sim_edge.findall("usage")
            ref_usages = ref_edge.findall("usage")

            for sim_usage, ref_usage in zip(sim_usages, ref_usages):
                usage_type = sim_usage.get("type")
                sim_percent = float(sim_usage.get("percent").rstrip("%"))
                ref_percent = float(ref_usage.get("percent").rstrip("%"))
                diff = abs(sim_percent - ref_percent)

                status = "✓" if diff < 0.1 else "!"
                print(
                    f"{status} Type {usage_type:8} - Sim: {sim_percent:5.1f}%, "
                    f"Ref: {ref_percent:5.1f}%, Diff: {diff:5.1f}%"
                )

                max_diff = max(max_diff, diff)
                total_diff += diff
                count += 1

        return max_diff, (total_diff / count if count > 0 else 0)

    def compare(self) -> bool:
        """Compare simulation results against reference data.

        Returns:
            bool: True if all checks pass, False otherwise
        """
        if not self.sim_path.exists() or not self.ref_path.exists():
            print(
                f"Error: Missing files\nSimulation: {self.sim_path}\nReference: {self.ref_path}"
            )
            return False

        try:
            # Parse both files
            sim_tree = ET.parse(self.sim_path)
            ref_tree = ET.parse(self.ref_path)

            # Compare delays and loads
            max_delay_diff, avg_delay_diff = self._compare_delays(sim_tree, ref_tree)
            max_load_diff, avg_load_diff = self._compare_loads(sim_tree, ref_tree)

            # Overall assessment
            delay_ok = max_delay_diff <= 1.1
            load_ok = max_load_diff <= 0.2

            print("\nOverall Status:", end=" ")
            if delay_ok and load_ok:
                print("✓ PASSED")
                return True
            else:
                print("! FAILED")
                return False

        except ET.ParseError as e:
            print(f"Error parsing XML files: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during comparison: {e}")
            return False


def main(file):
    # Parse network
    parser = Parser(Path(file))
    flows, targets, edges = parser.parse_network()

    print(edges)

    # Initial calculations
    network_calculus = NetworkCalculus(flows, targets, edges)
    network_calculus.compute()

    # Calculate final loads and save results
    writer = Writer(file)
    writer.write_results(list(flows.values()), edges)

    # Compare results
    checker = Check(file)
    checker.compare()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        xml_file = sys.argv[1]
    else:
        raise ValueError("You must provide an XML file")
    main(xml_file)

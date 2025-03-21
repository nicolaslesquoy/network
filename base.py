from __future__ import annotations
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# Constants
MODE = "DEBUG" # "PROD"
CAPACITY = 100 * 10**6  # 100 Mbps
PATH_TO_OUT = Path("out/")
assert PATH_TO_OUT.exists()


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
        self.objectif = 0.0  # Calculated at the end of parseXML
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


nodes: Dict[str, Node] = {}
edges: List[Edge] = []
flows: Dict[str, Flow] = {}
targets: List[Target] = []


def bytes_to_bits(byte: float) -> float:
    return byte * 8


def ms_to_s(milliseconds: float) -> float:
    return milliseconds / 1000


def find_edge(source: Node, dest: Node) -> int:
    found = [
        index
        for index in range(len(edges))
        if (source.name == edges[index].source.name)
        and (dest.name == edges[index].destination.name)
    ]
    assert len(found) == 1
    return found[0]


def parse_stations(root: ET.Element) -> None:
    for station in root.findall("station"):
        name = station.get("name")
        if name:
            nodes[name] = Station(name)


def parse_switches(root: ET.Element) -> None:
    for sw in root.findall("switch"):
        name = sw.get("name")
        if name:
            nodes[name] = Switch(name)


def parse_edges(root: ET.Element) -> None:
    for link in root.findall("link"):
        source_name = link.get("from")
        dest_name = link.get("to")
        name = link.get("name")

        if source_name and dest_name and name:
            source = nodes[source_name]
            dest = nodes[dest_name]

            edge = Edge(source, dest, name)
            edges.append(edge)
            # Create reverse edge
            edge_reverse = Edge(dest, source, name)
            edges.append(edge_reverse)


def parse_flows(root: ET.Element) -> None:
    for fl in root.findall("flow"):
        name = fl.get("name")
        source_name = fl.get("source")

        if name and source_name:
            source = nodes[source_name]
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
                flows[name] = flow

                edges_traversed: List[Edge] = []

                for tg in fl.findall("target"):
                    dest_name = tg.get("name")
                    if dest_name:
                        dest = nodes[dest_name]
                        target = Target(flow, dest)
                        flow.targets.append(target)

                        step_source = source
                        for pt in tg.findall("path"):
                            step_dest_name = pt.get("node")
                            if step_dest_name:
                                step_dest = nodes[step_dest_name]
                                edge = edges[find_edge(step_source, step_dest)]
                                target.path.append(edge)
                                if edge not in edges_traversed:
                                    edges_traversed.append(edge)
                                step_source = step_dest

                        targets.append(target)

                for edge in edges_traversed:
                    edge.objectif += 1


def parse_network(xml_file: str) -> Tuple[Dict[str, Flow], List[Target], List[Edge]]:
    path = Path(xml_file)
    if path.is_file():
        tree = ET.parse(path)
        root = tree.getroot()
        parse_stations(root)
        parse_switches(root)
        parse_edges(root)
        parse_flows(root)
        return flows, targets, edges
    else:
        print("File not found: " + xml_file)
        return {}, [], []


def s_to_ms(seconds: float) -> float:
    return seconds * 10**6


def save_network(xml_file, flows, edges):
    path = Path(xml_file)
    suffix = "_DEBUG" if MODE == "DEBUG" else ""
    res_file = PATH_TO_OUT / f"{path.stem}_res{suffix}{path.suffix}"
    with res_file.open('w') as res:
        res.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        res.write("<results>\n")
        # Les delays
        res.write("\t<delays>\n")
        for flow in flows:
            res.write('\t\t<flow name="' + flow.name + '">\n')
            for target in flow.targets:
                res.write(
                    f'\t\t\t<target name="{target.destination.name}" value="{s_to_ms(target.total_delay)}" />\n'
                )
            res.write("\t\t</flow>\n")
        res.write("\t</delays>\n")

        res.write("\t<load>\n")
        for i in range(0, len(edges) - 1, 2):
            edgeDirect = edges[i]
            edgeReverse = edges[i + 1]
            res.write('\t\t<edge name="' + edgeDirect.name + '">\n')
            res.write(
                f'\t\t\t<usage percent="{edgeDirect.load:.1f}%" type="direct" value="{edgeDirect.arrival_curve_aggregated.rate}" />\n'
            )
            res.write(
                f'\t\t\t<usage percent="{edgeReverse.load:.1f}%" type="reverse" value="{edgeReverse.arrival_curve_aggregated.rate}" />\n'
            )
            res.write("\t\t</edge>\n")
        res.write("\t</load>\n")

        res.write("</results>\n")
    
    # print_output(str(res_file))

def print_output(file):
    path = Path(file)
    with path.open('r') as f:
        for line in f:
            print(line.rstrip())


def add_curve_to_switch(target, edge):
    already_accounted = target.flow.name in edge.flows_passed
    if not already_accounted:
        edge.arrival_curve_aggregated.add(
            target.arrivalCurve.burst, target.arrivalCurve.rate
        )
        edge.flows_passed.append(target.flow.name)


def check_proceed(target):
    if target.completed:
        return False
    edge = target.path[target.current_step]
    return len(edge.flows_passed) == edge.objectif


def calculate_station_arrival_curves(flows):
    """Calculate initial arrival curves for each source station"""
    for flow in flows.values():
        flow.source.arrival_curve_aggregated.add(
            flow.get_data_length(), flow.get_rate()
        )

def process_source_stations(targets):
    """Process initial delays from source stations"""
    for target in targets:
        source = target.flow.source
        edge = target.path[target.current_step]
        
        # Add source station's curve to first edge
        add_curve_to_switch(target, edge)
        
        # Calculate source delay if needed
        if edge.delay == 0:
            edge.delay = source.arrival_curve_aggregated.burst / CAPACITY
            
        # Update arrival curve and move to next step
        target.arrivalCurve.add_delay(edge.delay)
        target.current_step += 1
        
        # Add curve to next switch
        next_edge = target.path[target.current_step]
        add_curve_to_switch(target, next_edge)

def process_target(target):
    """Process a single target through its path"""
    while check_proceed(target):
        edge = target.path[target.current_step]
        
        # Calculate edge delay if needed
        if edge.delay == 0:
            edge.delay = edge.arrival_curve_aggregated.burst / CAPACITY
            
        target.arrivalCurve.add_delay(edge.delay)

        # Check if target reached destination
        if target.path[target.current_step] == target.path[-1]:
            target.total_delay = sum(edge.delay for edge in target.path)
            target.completed = True
            return True
            
        # Move to next edge
        target.current_step += 1
        next_edge = target.path[target.current_step]
        add_curve_to_switch(target, next_edge)
    
    return False

def calculate_edge_loads(edges):
    """Calculate final load percentages for all edges
    
    For each edge:
    1. Calculate direct load based on aggregated arrival curve rate
    2. Express loads as percentages of link capacity
    """
    for i in range(0, len(edges), 2):
        direct_edge = edges[i]

        print(f"Processing edge {direct_edge.name}")
        print(f"Arrival curve: {direct_edge.arrival_curve_aggregated.burst:.1f} bytes, {direct_edge.arrival_curve_aggregated.rate:.1f} bps")
        # Calculate direct traffic load
        direct_edge.load = direct_edge.arrival_curve_aggregated.rate * 100 / CAPACITY
        
        # Ensure loads don't exceed link capacity
        total_load = direct_edge.load
        if total_load > 100:
            print(f"Warning: Total load {total_load:.1f}% exceeds link capacity for {direct_edge.name}")

def main(file):
    # Parse network
    flows, targets, edges = parse_network(file)

    print(edges)
    
    # Initial calculations
    calculate_station_arrival_curves(flows)
    process_source_stations(targets)

    # Process all targets
    targets_completed = 0
    current_index = 0
    
    while targets_completed < len(targets):
        target = targets[current_index]
        if not target.completed and process_target(target):
            targets_completed += 1
        current_index = (current_index + 1) % len(targets)

    # Calculate final loads and save results
    calculate_edge_loads(edges)
    save_network(file, flows.values(), edges)

def compare_results(input_file: str) -> None:
    """Compare simulation results with reference results and generate a report.
    
    Compares delays and loads between simulation output and reference files,
    generating a detailed comparison report with statistics.
    
    Args:
        input_file: Path to the input XML file used for simulation
    """
    def compare_delays(sim_tree: ET.ElementTree, ref_tree: ET.ElementTree) -> tuple[float, float]:
        """Compare delays between simulation and reference results."""
        print("\n=== Delay Comparison ===")
        print("------------------------")
        
        sim_delays = sim_tree.findall(".//flow")
        ref_delays = ref_tree.findall(".//flow")
        
        max_diff = 0.0
        total_diff = 0.0
        count = 0
        
        for sim_flow, ref_flow in zip(sim_delays, ref_delays):
            flow_name = sim_flow.get('name')
            print(f"\nFlow: {flow_name}")
            print("-" * (len(flow_name) + 7))
            
            sim_targets = sim_flow.findall('target')
            ref_targets = ref_flow.findall('target')
            
            for sim_target, ref_target in zip(sim_targets, ref_targets):
                target_name = sim_target.get('name')
                sim_delay = float(sim_target.get('value'))
                ref_delay = float(ref_target.get('value'))
                diff = abs(sim_delay - ref_delay)
                
                status = "✓" if diff < 1.0 else "!"
                print(f"{status} Target {target_name:10} - Sim: {sim_delay:8.1f}μs, Ref: {ref_delay:8.1f}μs, Diff: {diff:8.1f}μs")
                
                max_diff = max(max_diff, diff)
                total_diff += diff
                count += 1
        
        return max_diff, (total_diff / count if count > 0 else 0)

    def compare_loads(sim_tree: ET.ElementTree, ref_tree: ET.ElementTree) -> tuple[float, float]:
        """Compare loads between simulation and reference results."""
        print("\n=== Load Comparison ===")
        print("----------------------")
        
        sim_edges = sim_tree.findall(".//edge")
        ref_edges = ref_tree.findall(".//edge")
        
        max_diff = 0.0
        total_diff = 0.0
        count = 0
        
        for sim_edge, ref_edge in zip(sim_edges, ref_edges):
            edge_name = sim_edge.get('name')
            print(f"\nEdge: {edge_name}")
            print("-" * (len(edge_name) + 7))
            
            sim_usages = sim_edge.findall('usage')
            ref_usages = ref_edge.findall('usage')
            
            for sim_usage, ref_usage in zip(sim_usages, ref_usages):
                usage_type = sim_usage.get('type')
                sim_percent = float(sim_usage.get('percent').rstrip('%'))
                ref_percent = float(ref_usage.get('percent').rstrip('%'))
                diff = abs(sim_percent - ref_percent)
                
                status = "✓" if diff < 0.1 else "!"
                print(f"{status} Type {usage_type:8} - Sim: {sim_percent:5.1f}%, Ref: {ref_percent:5.1f}%, Diff: {diff:5.1f}%")
                
                max_diff = max(max_diff, diff)
                total_diff += diff
                count += 1
        
        return max_diff, (total_diff / count if count > 0 else 0)

    # Get paths for both files
    sim_path = PATH_TO_OUT / f"{Path(input_file).stem}_res{'_DEBUG' if MODE=='DEBUG' else ''}.xml"
    ref_path = Path(input_file).parent / f"{Path(input_file).stem}_res.xml"
    
    if not sim_path.exists() or not ref_path.exists():
        print(f"Error: Missing files\nSimulation: {sim_path}\nReference: {ref_path}")
        return

    try:
        # Parse both files
        sim_tree = ET.parse(sim_path)
        ref_tree = ET.parse(ref_path)
        
        # Compare delays and loads
        max_delay_diff, avg_delay_diff = compare_delays(sim_tree, ref_tree)
        max_load_diff, avg_load_diff = compare_loads(sim_tree, ref_tree)
        
        # Overall assessment
        delay_ok = max_delay_diff < 1.0 and avg_delay_diff < 0.5
        load_ok = max_load_diff < 0.1 and avg_load_diff < 0.05
        
        print("\nOverall Status:", end=" ")
        if delay_ok and load_ok:
            print("✓ PASSED")
        else:
            print("! FAILED")
            
    except ET.ParseError as e:
        print(f"Error parsing XML files: {e}")
    except Exception as e:
        print(f"Unexpected error during comparison: {e}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        xml_file = sys.argv[1]
    else:
        raise ValueError("You must provide an XML file")
    main(xml_file)
    compare_results(xml_file)
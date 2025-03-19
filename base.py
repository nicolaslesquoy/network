from __future__ import annotations
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import pathlib
import sys

# Global variables
CAPACITY = 100 * 10**6
OVERHEAD = 67  # bytes
DECIMAL_PRECISION = 6  # microsecond precision


# Utility functions
def bytes_to_bits(byte: float) -> float:
    return byte * 8


def ms_to_s(milliseconds: float) -> float:
    return milliseconds / 1000


def s_to_ms(seconds: float) -> float:
    return seconds * 10**6


def print_output(file):
    with open(file, "r") as f:
        for line in f:
            print(line.rstrip())

class ArrivalCurve:
    def __init__(self, burst: float, rate: float) -> None:
        self.burst = burst
        self.rate = rate

    def __add__(self, burst: float, rate: float) -> None:
        self.burst += burst
        self.rate += rate

    def __repr__(self) -> str:
        return f"ArrivalCurve({self.burst},{self.rate})"
    
    def __str__(self) -> str:
        return f"ArrivalCurve with burst {self.burst} and rate {self.rate}"
    
    def add_delay(self, delay: float) -> None:
        self.burst += delay * self.rate

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
        return self.get_data_length() / ms_to_s(self.period)

    def __repr__(self) -> str:
        return f"Flow {self.name}"

    def __str__(self) -> str:
        return f"Flow {self.name}"

class Target:
    def __init__(self, flow: "Flow", destination: "Station") -> None:
        self.flow = flow
        self.destination = destination
        self.path: list["Node"] = []
    
    def __repr__(self) -> str:
        return f"Target in {self.flow.name} to {self.destination.name} with path {self.path}"
    
    def __str__(self) -> str:
        return f"Target in {self.flow.name} to {self.destination.name} with path {self.path}"
    
class Link:
    def __init__(self, name: str, source: "Node", destination: "Node", to_port: int, from_port: int, capacity: int) -> None:
        self.name = name
        self.source = source
        self.destination = destination
        self.to_port = to_port
        self.from_port = from_port
        self.capacity = capacity

    def __repr__(self) -> str:
        return f"Link from {self.source.name} to {self.destination.name}"

    def __str__(self) -> str:
        return f"Link from {self.source.name} to {self.destination.name}"
    
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
        super().__init__(name)

class Parser:
    def __init__(self, file: pathlib.Path) -> None:
        self.file = file
        self.tree = ET.parse(str(file))
        self.root = self.tree.getroot()
        
        # Initialize collections as class attributes
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        self.flows: Dict[str, Flow] = {}
        self.targets: List[Target] = []

    def parse_station(self) -> None:
        for station in self.root.findall('station'):
            self.nodes[station.get('name')] = Station(station.get('name'))

    def parse_switch(self) -> None:
        for switch in self.root.findall('switch'):
            self.nodes[switch.get('name')] = Switch(switch.get('name'))
        
    def parse_link(self) -> None:
        for link in self.root.findall('link'):
            source = self.nodes[link.get('from')]
            destination = self.nodes[link.get('to')]
            self.links.append(Link(link.get('name'), source, destination, link.get('toPort'), link.get('fromPort'), link.get('transmission-capacity')))
    
    def parse_flow(self) -> None:
        for flow_elem in self.root.findall('flow'):
            source = self.nodes[flow_elem.get('source')]
            name = flow_elem.get('name')
            
            # Create flow
            flow = Flow(
                name,
                OVERHEAD,
                float(flow_elem.get('max-payload')),
                float(flow_elem.get('period')),
                source
            )
            self.flows[name] = flow
            
            # Parse targets for this flow
            for target_elem in flow_elem.findall('target'):
                destination = self.nodes[target_elem.get('name')]
                target = Target(flow, destination)
                
                # Extract path nodes
                path = [flow.source]
                for path_elem in target_elem.findall('path'):
                    node_name = path_elem.get('node')
                    path.append(self.nodes[node_name])
                
                target.path = path
                flow.targets.append(target)
                self.targets.append(target)

    def parse(self) -> None:
        """Parse the XML file and create corresponding objects"""
        self.parse_station()
        self.parse_switch()
        self.parse_link()
        self.parse_flow()

    def debug(self) -> str:
        return f"Nodes: {self.nodes}\nLinks: {self.links}\nFlows: {self.flows}\nTargets: {self.targets}"
    
    def __repr__(self) -> str:
        return f"Parser for {self.file}"
    
    def __str__(self) -> str:
        return f"Parser for {self.file}"
    
class NetworkCalculus:
    None

if __name__ == "__main__":
    file = pathlib.Path(sys.argv[1])
    parser = Parser(file)
    parser.parse()
    print(parser.debug())
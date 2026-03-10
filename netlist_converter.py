#!/usr/bin/env python3
"""
自动识别并转换电路网表为Phy-Engine可模拟的格式
支持从多种输入格式转换为Phy-Engine的元件代码、连线和属性数组
"""

import re
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Component:
    """电路元件数据类"""
    name: str
    element_code: int
    properties: List[float]
    pins: List[str]


class NetlistConverter:
    """网表转换器，将各种格式的电路描述转换为Phy-Engine可识别的格式"""
    
    # Phy-Engine元件代码映射
    ELEMENT_CODES = {
        'R': 1,    # 电阻
        'C': 2,    # 电容
        'L': 3,    # 电感
        'V': 4,    # 直流电压源
        'I': 6,    # 直流电流源
        'Q': 50,   # BJT NPN
        'QPNP': 51, # BJT PNP
        'M': 52,   # MOSFET NMOS
        'MP': 53,  # MOSFET PMOS
        'D': 13,   # 二极管
        'E': 9,    # VCVS
        'G': 8,    # VCCS
        'H': 11,   # CCVS
        'F': 10,   # CCCS
        'SW': 12,  # 开关
        'X': 200,  # 数字输入
        'Y': 201,  # 数字输出
        'AND': 204,   # AND门
        'OR': 202,    # OR门
        'NOT': 205,   # NOT门
        'NAND': 208,  # NAND门
        'NOR': 209,   # NOR门
        'XOR': 206,   # XOR门
        'XNOR': 207,  # XNOR门
        'BUF': 203,   # 缓冲器
        'HA': 220,    # 半加器
        'FA': 221,    # 全加器
        'DFF': 225,   # D触发器
        'JKFF': 228,  # JK触发器
        'TFF': 226,   # T触发器
        'COUNTER': 229,  # 计数器
        'RAM': 232,   # RAM
        'ADC': 231,   # ADC
        'DAC': 232,   # DAC
    }
    
    def __init__(self):
        self.components: List[Component] = []
        self.connections: List[Tuple[int, int, int, int]] = []  # [ele1, pin1, ele2, pin2]
        self.node_map: Dict[str, int] = {'0': 0, 'GND': 0, 'gnd': 0}  # 节点名称到节点ID的映射
        self.next_node_id = 1
        self.next_element_id = 0
    
    def parse_spice_netlist(self, netlist: str) -> None:
        """解析SPICE格式网表"""
        lines = netlist.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            # 解析元件行
            parts = line.split()
            if not parts:
                continue
            
            name = parts[0].upper()
            if name.startswith('R'):  # 电阻
                self._parse_resistor(name, parts[1:])
            elif name.startswith('C'):  # 电容
                self._parse_capacitor(name, parts[1:])
            elif name.startswith('L'):  # 电感
                self._parse_inductor(name, parts[1:])
            elif name.startswith('V'):  # 电压源
                self._parse_voltage_source(name, parts[1:])
            elif name.startswith('I'):  # 电流源
                self._parse_current_source(name, parts[1:])
            elif name.startswith('Q'):  # BJT
                self._parse_bjt(name, parts[1:])
            elif name.startswith('M'):  # MOSFET
                self._parse_mosfet(name, parts[1:])
            elif name.startswith('D'):  # 二极管
                self._parse_diode(name, parts[1:])
            elif name.startswith('E'):  # VCVS
                self._parse_vcvs(name, parts[1:])
            elif name.startswith('G'):  # VCCS
                self._parse_vccs(name, parts[1:])
            elif name.startswith('H'):  # CCVS
                self._parse_ccvs(name, parts[1:])
            elif name.startswith('F'):  # CCCS
                self._parse_cccs(name, parts[1:])
            elif name.startswith('SW'):  # 开关
                self._parse_switch(name, parts[1:])
    
    def _parse_resistor(self, name: str, params: List[str]) -> None:
        """解析电阻"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        value_str = params[2]
        
        # 解析电阻值
        resistance = self._parse_value(value_str)
        
        # 添加元件
        comp = Component(name, self.ELEMENT_CODES['R'], [resistance], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_capacitor(self, name: str, params: List[str]) -> None:
        """解析电容"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        value_str = params[2]
        
        # 解析电容值
        capacitance = self._parse_value(value_str)
        
        # 添加元件
        comp = Component(name, self.ELEMENT_CODES['C'], [capacitance], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_inductor(self, name: str, params: List[str]) -> None:
        """解析电感"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        value_str = params[2]
        
        # 解析电感值
        inductance = self._parse_value(value_str)
        
        # 添加元件
        comp = Component(name, self.ELEMENT_CODES['L'], [inductance], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_voltage_source(self, name: str, params: List[str]) -> None:
        """解析电压源"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        value_str = params[2]
        
        # 解析电压值
        voltage = self._parse_value(value_str)
        
        # 添加元件
        comp = Component(name, self.ELEMENT_CODES['V'], [voltage], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_current_source(self, name: str, params: List[str]) -> None:
        """解析电流源"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        value_str = params[2]
        
        # 解析电流值
        current = self._parse_value(value_str)
        
        # 添加元件
        comp = Component(name, self.ELEMENT_CODES['I'], [current], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_bjt(self, name: str, params: List[str]) -> None:
        """解析BJT"""
        if len(params) < 4:
            return
        
        collector, base, emitter = params[0], params[1], params[2]
        model = params[3] if len(params) > 3 else "default"
        
        # 简化处理，使用默认参数
        comp = Component(name, self.ELEMENT_CODES['Q'], [1e-14, 1.0, 100.0, 300.0, 1.0], [collector, base, emitter])
        self.components.append(comp)
        
        # 添加连接
        collector_id = self._get_node_id(collector)
        base_id = self._get_node_id(base)
        emitter_id = self._get_node_id(emitter)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 1, self.next_element_id, 2))
        self.next_element_id += 1
    
    def _parse_mosfet(self, name: str, params: List[str]) -> None:
        """解析MOSFET"""
        if len(params) < 5:
            return
        
        drain, gate, source, bulk = params[0], params[1], params[2], params[3]
        model = params[4] if len(params) > 4 else "default"
        
        # 简化处理，使用默认参数
        comp = Component(name, self.ELEMENT_CODES['M'], [1e-4, 0.01, 1.0], [drain, gate, source, bulk])
        self.components.append(comp)
        
        # 添加连接
        drain_id = self._get_node_id(drain)
        gate_id = self._get_node_id(gate)
        source_id = self._get_node_id(source)
        bulk_id = self._get_node_id(bulk)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 1, self.next_element_id, 2))
        self.connections.append((self.next_element_id, 2, self.next_element_id, 3))
        self.next_element_id += 1
    
    def _parse_diode(self, name: str, params: List[str]) -> None:
        """解析二极管"""
        if len(params) < 3:
            return
        
        anode, cathode = params[0], params[1]
        model = params[2] if len(params) > 2 else "default"
        
        # 简化处理，使用默认参数
        comp = Component(name, self.ELEMENT_CODES['D'], [1e-14, 1.0, 0, 1.0, 300.0, 0, 0, 0, 1.0], [anode, cathode])
        self.components.append(comp)
        
        # 添加连接
        anode_id = self._get_node_id(anode)
        cathode_id = self._get_node_id(cathode)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _parse_vcvs(self, name: str, params: List[str]) -> None:
        """解析VCVS"""
        if len(params) < 5:
            return
        
        out_plus, out_minus, in_plus, in_minus = params[0], params[1], params[2], params[3]
        gain = float(params[4]) if len(params) > 4 else 1.0
        
        comp = Component(name, self.ELEMENT_CODES['E'], [gain], [out_plus, out_minus, in_plus, in_minus])
        self.components.append(comp)
        
        # 添加连接
        out_plus_id = self._get_node_id(out_plus)
        out_minus_id = self._get_node_id(out_minus)
        in_plus_id = self._get_node_id(in_plus)
        in_minus_id = self._get_node_id(in_minus)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 2, self.next_element_id, 3))
        self.next_element_id += 1
    
    def _parse_vccs(self, name: str, params: List[str]) -> None:
        """解析VCCS"""
        if len(params) < 5:
            return
        
        out_plus, out_minus, in_plus, in_minus = params[0], params[1], params[2], params[3]
        gain = float(params[4]) if len(params) > 4 else 1.0
        
        comp = Component(name, self.ELEMENT_CODES['G'], [gain], [out_plus, out_minus, in_plus, in_minus])
        self.components.append(comp)
        
        # 添加连接
        out_plus_id = self._get_node_id(out_plus)
        out_minus_id = self._get_node_id(out_minus)
        in_plus_id = self._get_node_id(in_plus)
        in_minus_id = self._get_node_id(in_minus)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 2, self.next_element_id, 3))
        self.next_element_id += 1
    
    def _parse_ccvs(self, name: str, params: List[str]) -> None:
        """解析CCVS"""
        if len(params) < 5:
            return
        
        out_plus, out_minus, in_plus, in_minus = params[0], params[1], params[2], params[3]
        gain = float(params[4]) if len(params) > 4 else 1.0
        
        comp = Component(name, self.ELEMENT_CODES['H'], [gain], [out_plus, out_minus, in_plus, in_minus])
        self.components.append(comp)
        
        # 添加连接
        out_plus_id = self._get_node_id(out_plus)
        out_minus_id = self._get_node_id(out_minus)
        in_plus_id = self._get_node_id(in_plus)
        in_minus_id = self._get_node_id(in_minus)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 2, self.next_element_id, 3))
        self.next_element_id += 1
    
    def _parse_cccs(self, name: str, params: List[str]) -> None:
        """解析CCCS"""
        if len(params) < 5:
            return
        
        out_plus, out_minus, in_plus, in_minus = params[0], params[1], params[2], params[3]
        gain = float(params[4]) if len(params) > 4 else 1.0
        
        comp = Component(name, self.ELEMENT_CODES['F'], [gain], [out_plus, out_minus, in_plus, in_minus])
        self.components.append(comp)
        
        # 添加连接
        out_plus_id = self._get_node_id(out_plus)
        out_minus_id = self._get_node_id(out_minus)
        in_plus_id = self._get_node_id(in_plus)
        in_minus_id = self._get_node_id(in_minus)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.connections.append((self.next_element_id, 2, self.next_element_id, 3))
        self.next_element_id += 1
    
    def _parse_switch(self, name: str, params: List[str]) -> None:
        """解析开关"""
        if len(params) < 3:
            return
        
        node1, node2 = params[0], params[1]
        model = params[2] if len(params) > 2 else "default"
        
        # 简化处理，开关默认为断开状态
        comp = Component(name, self.ELEMENT_CODES['SW'], [0.0], [node1, node2])
        self.components.append(comp)
        
        # 添加连接
        node1_id = self._get_node_id(node1)
        node2_id = self._get_node_id(node2)
        self.connections.append((self.next_element_id, 0, self.next_element_id, 1))
        self.next_element_id += 1
    
    def _get_node_id(self, node_name: str) -> int:
        """获取节点ID，如果不存在则创建新ID"""
        if node_name not in self.node_map:
            self.node_map[node_name] = self.next_node_id
            self.next_node_id += 1
        return self.node_map[node_name]
    
    def _parse_value(self, value_str: str) -> float:
        """解析带单位的数值"""
        value_str = value_str.upper()
        
        # 单位映射
        units = {
            'T': 1e12,
            'G': 1e9,
            'MEG': 1e6,
            'K': 1e3,
            'M': 1e-3,
            'U': 1e-6,
            'N': 1e-9,
            'P': 1e-12,
            'F': 1e-15
        }
        
        # 查找单位
        for unit, multiplier in units.items():
            if value_str.endswith(unit):
                num_part = value_str[:-len(unit)]
                try:
                    return float(num_part) * multiplier
                except ValueError:
                    return 0.0
        
        # 如果没有单位，直接转换为浮点数
        try:
            return float(value_str)
        except ValueError:
            return 0.0
    
    def convert_to_phy_engine_format(self) -> Tuple[List[int], List[int], List[float]]:
        """转换为Phy-Engine可识别的格式"""
        elements = []
        properties = []
        
        # 构建元件数组和属性数组
        for comp in self.components:
            elements.append(comp.element_code)
            properties.extend(comp.properties)
        
        # 构建连线数组
        wires = []
        for conn in self.connections:
            ele1, pin1, ele2, pin2 = conn
            wires.extend([ele1, pin1, ele2, pin2])
        
        return elements, wires, properties
    
    def parse_verilog(self, verilog_code: str) -> None:
        """解析Verilog代码并转换为数字电路网表"""
        # 简单的Verilog解析器，识别基本门电路和模块实例
        lines = verilog_code.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            # 解析门实例
            if 'and ' in line or line.startswith('and '):
                self._parse_verilog_gate(line, 'AND')
            elif 'or ' in line or line.startswith('or '):
                self._parse_verilog_gate(line, 'OR')
            elif 'not ' in line or line.startswith('not '):
                self._parse_verilog_gate(line, 'NOT')
            elif 'nand ' in line or line.startswith('nand '):
                self._parse_verilog_gate(line, 'NAND')
            elif 'nor ' in line or line.startswith('nor '):
                self._parse_verilog_gate(line, 'NOR')
            elif 'xor ' in line or line.startswith('xor '):
                self._parse_verilog_gate(line, 'XOR')
            elif 'xnor ' in line or line.startswith('xnor '):
                self._parse_verilog_gate(line, 'XNOR')
            elif 'buf ' in line or line.startswith('buf '):
                self._parse_verilog_gate(line, 'BUF')
    
    def _parse_verilog_gate(self, line: str, gate_type: str) -> None:
        """解析Verilog门实例"""
        # 简化处理，提取输入输出端口
        # 格式: and name (out, in1, in2);
        # 或: and (out, in1, in2);
        
        # 移除注释和多余空格
        line = re.sub(r'//.*', '', line)
        line = re.sub(r'/\*.*?\*/', '', line)
        line = line.replace(';', '').strip()
        
        # 提取端口
        if '(' in line and ')' in line:
            ports_part = line[line.find('(')+1:line.find(')')]
            ports = [p.strip() for p in ports_part.split(',')]
            
            if len(ports) >= 2:
                out_port = ports[0]
                in_ports = ports[1:]
                
                # 创建门电路元件
                comp = Component(f"{gate_type}_{len(self.components)}", 
                                self.ELEMENT_CODES[gate_type], 
                                [], 
                                [out_port] + in_ports)
                self.components.append(comp)
                
                # 添加连接
                out_id = self._get_node_id(out_port)
                for i, in_port in enumerate(in_ports):
                    in_id = self._get_node_id(in_port)
                    self.connections.append((self.next_element_id, 0, self.next_element_id, i+1))
                
                self.next_element_id += 1
    
    def parse_json_netlist(self, json_netlist: Dict[str, Any]) -> None:
        """解析JSON格式的网表"""
        if 'components' in json_netlist:
            for comp_data in json_netlist['components']:
                comp_type = comp_data.get('type', '').upper()
                name = comp_data.get('name', f'comp_{len(self.components)}')
                nodes = comp_data.get('nodes', [])
                params = comp_data.get('params', [])
                
                if comp_type in self.ELEMENT_CODES:
                    element_code = self.ELEMENT_CODES[comp_type]
                    properties = [float(p) for p in params if isinstance(p, (int, float))]
                    
                    comp = Component(name, element_code, properties, nodes)
                    self.components.append(comp)
                    
                    # 添加连接
                    for i in range(len(nodes)-1):
                        node1_id = self._get_node_id(nodes[i])
                        node2_id = self._get_node_id(nodes[i+1])
                        self.connections.append((self.next_element_id, i, self.next_element_id, i+1))
                    
                    self.next_element_id += 1


def convert_netlist(input_data: str, input_format: str = 'spice') -> Tuple[List[int], List[int], List[float]]:
    """
    将输入的网表转换为Phy-Engine可识别的格式
    
    Args:
        input_data: 输入的网表数据
        input_format: 输入格式 ('spice', 'verilog', 'json')
    
    Returns:
        Tuple[List[int], List[int], List[float]]: (elements, wires, properties)
    """
    converter = NetlistConverter()
    
    if input_format.lower() == 'spice':
        converter.parse_spice_netlist(input_data)
    elif input_format.lower() == 'verilog':
        converter.parse_verilog(input_data)
    elif input_format.lower() == 'json':
        try:
            json_data = json.loads(input_data)
            converter.parse_json_netlist(json_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    return converter.convert_to_phy_engine_format()


# 示例使用
if __name__ == "__main__":
    # 示例SPICE网表
    spice_netlist = """
    * Simple RC Circuit
    V1 in 0 DC 5
    R1 in out 1k
    C1 out 0 1uF
    .END
    """
    
    # 转换为Phy-Engine格式
    elements, wires, properties = convert_netlist(spice_netlist, 'spice')
    
    print("Elements:", elements)
    print("Wires:", wires)
    print("Properties:", properties)
    
    # 示例Verilog代码
    verilog_code = """
    module test_and (out, a, b);
        input a, b;
        output out;
        and gate1 (out, a, b);
    endmodule
    """
    
    elements_v, wires_v, properties_v = convert_netlist(verilog_code, 'verilog')
    
    print("\nVerilog Elements:", elements_v)
    print("Verilog Wires:", wires_v)
    print("Verilog Properties:", properties_v)
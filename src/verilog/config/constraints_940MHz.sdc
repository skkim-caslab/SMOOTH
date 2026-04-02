# 클럭 정의 (1: 1GHz, 주기 1ns)
# 클럭 정의 (10: 100MHz, 주기 10ns)
# 클럭 정의 (1.0638: 940MHz, 주기 1.0638ns)
create_clock -name clk -period 1.0638 [get_ports clk]
set_output_delay -clock clk 2 [all_outputs]
set_load 1 [all_outputs]

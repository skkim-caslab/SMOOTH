read_liberty ../libs/asap7.lib
read_verilog synthesized_address_check.v
link_design address_check
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/energy/verilog/address_check.out
report_power >> ../../../data/energy/verilog/address_check.out

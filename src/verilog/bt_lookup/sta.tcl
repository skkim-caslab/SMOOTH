read_liberty ../libs/asap7.lib
read_verilog synthesized_bt_lookup.v
link_design bt_lookup
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/energy/verilog/bt_lookup.out
report_power >> ../../../data/energy/verilog/bt_lookup.out

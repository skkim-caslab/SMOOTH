module bt_lookup (
    input clk, rst_n,
    input [13:0] addr,          // Input address (virtual)
    input lookup_flg,           // Lookup flag
    input end_cmd,              // End command
    input [31:0] sram_data_out, // SRAM data output (for data)
    input [28:0] sram_bt_data_out, // SRAM block table data output
    output [31:0] data,         // Output data
    output reg [12:0] p_blk,    // Physical block address
    output reg [12:0] cont,     // Continuity
    output reg [12:0] sram_addr,// SRAM address (for data)
    output reg sram_ce,         // SRAM chip enable (for data)
    output reg [13:0] sram_bt_addr, // SRAM block table address
    output reg [28:0] sram_bt_data_in, // SRAM block table data input
    output reg sram_bt_ce,      // SRAM block table chip enable
    output reg sram_bt_we       // SRAM block table write enable
);

    reg [13:0] index;           // Current index for table initialization
    reg [2:0] state, next_state; // FSM states
    reg [28:0] block_table_entry; // Temporary storage for block table entry
    assign data = sram_data_out;

    localparam IDLE = 3'b000,
               RESET_TABLE = 3'b001,
               PROCESS = 3'b010,
               READ_SRAM = 3'b011,
               UPDATE_SRAM = 3'b100;

    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESET_TABLE;
            p_blk <= 0;
            cont <= 0;
            sram_addr <= 0;
            sram_ce <= 0;
            sram_bt_addr <= 0;
            sram_bt_ce <= 0;
            sram_bt_we <= 0;
            sram_bt_data_in <= 0;
            index <= 0;
            block_table_entry <= 0;
        end else begin
            state <= next_state;
            case (state)
                RESET_TABLE: begin
                    sram_bt_addr <= index;
                    sram_bt_data_in <= 0;
                    sram_bt_ce <= 1;
                    sram_bt_we <= 1;
                    if (index < 16383) begin
                        index <= index + 1;
                    end else begin
                        index <= 0;
                        sram_bt_ce <= 0;
                        sram_bt_we <= 0;
                    end
                end
                PROCESS: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                    if (lookup_flg) begin
                        sram_bt_addr <= addr;
                        sram_bt_ce <= 1;
                        sram_bt_we <= 0; // Read mode
                    end
                end
                READ_SRAM: begin
                    sram_bt_ce <= 0;
                    block_table_entry <= sram_bt_data_out;
                    p_blk <= sram_bt_data_out[28:16];
                    cont <= sram_bt_data_out[15:3];
                    sram_addr <= sram_bt_data_out[28:16];
                    sram_ce <= 1;
                    if (end_cmd && sram_bt_data_out[2:0] > 0) begin
                        sram_bt_data_in <= {sram_bt_data_out[28:3], sram_bt_data_out[2:0] - 1};
                        sram_bt_addr <= addr;
                        sram_bt_ce <= 1;
                        sram_bt_we <= 1;
                    end
                end
                UPDATE_SRAM: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                end
            endcase
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            RESET_TABLE: begin
                if (index == 16383) next_state = IDLE;
            end
            IDLE: begin
                if (lookup_flg) next_state = PROCESS;
            end
            PROCESS: begin
                next_state = READ_SRAM;
            end
            READ_SRAM: begin
                if (end_cmd && block_table_entry[2:0] > 0) next_state = UPDATE_SRAM;
                else next_state = IDLE;
            end
            UPDATE_SRAM: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
endmodule

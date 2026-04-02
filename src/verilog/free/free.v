module free (
    input clk, rst_n,
    input [28:0] sram_bt_data_out, // SRAM block table data output
    input [31:0] sram_bm_data_out, // SRAM bitmap data output
    output reg done,               // Scan cycle complete
    output reg [13:0] sram_bt_addr, // SRAM block table address
    output reg [28:0] sram_bt_data_in, // SRAM block table data input
    output reg sram_bt_ce,         // SRAM block table chip enable
    output reg sram_bt_we,         // SRAM block table write enable
    output reg [7:0] sram_bm_addr, // SRAM bitmap address (256 entries for 8192 bits)
    output reg [31:0] sram_bm_data_in, // SRAM bitmap data input
    output reg sram_bm_ce,         // SRAM bitmap chip enable
    output reg sram_bm_we          // SRAM bitmap write enable
);

    reg [13:0] scan_idx;           // Scan index
    reg [28:0] table_entry;        // Temporary register for block table read
    reg [31:0] bitmap_entry;       // Temporary register for bitmap read
    reg bitmap_bit;                // Specific bit from bitmap entry
    reg [12:0] alloc_start_addr;   // Bitmap address for bit access
    reg [4:0] bitmap_bit_idx;      // Bit index within 32-bit bitmap entry
    reg [3:0] state, next_state;   // FSM states

    localparam IDLE = 4'b0000,
               RESET_SRAM = 4'b0001,
               SCAN_READ_BT = 4'b0010,
               SCAN_READ_BM = 4'b0011,
               SCAN_PROCESS = 4'b0100,
               SCAN_UPDATE_BT = 4'b0101,
               SCAN_UPDATE_BM = 4'b0110;

    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESET_SRAM;
            scan_idx <= 0;
            table_entry <= 0;
            bitmap_entry <= 0;
            bitmap_bit <= 0;
            alloc_start_addr <= 0;
            bitmap_bit_idx <= 0;
            done <= 0;
            sram_bt_addr <= 0;
            sram_bt_data_in <= 0;
            sram_bt_ce <= 0;
            sram_bt_we <= 0;
            sram_bm_addr <= 0;
            sram_bm_data_in <= 0;
            sram_bm_ce <= 0;
            sram_bm_we <= 0;
        end else begin
            state <= next_state;
            case (state)
                RESET_SRAM: begin
                    sram_bt_addr <= scan_idx;
                    sram_bt_data_in <= 0;
                    sram_bt_ce <= 1;
                    sram_bt_we <= 1;
                    sram_bm_addr <= scan_idx[7:0]; // Initialize bitmap SRAM
                    sram_bm_data_in <= 0;
                    sram_bm_ce <= 1;
                    sram_bm_we <= 1;
                    if (scan_idx < 16383) begin
                        scan_idx <= scan_idx + 1;
                    end else begin
                        scan_idx <= 0;
                        sram_bt_ce <= 0;
                        sram_bt_we <= 0;
                        sram_bm_ce <= 0;
                        sram_bm_we <= 0;
                    end
                end
                SCAN_READ_BT: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                    sram_bt_addr <= scan_idx;
                    sram_bt_ce <= 1;
                    sram_bt_we <= 0; // Read block table
                end
                SCAN_READ_BM: begin
                    sram_bt_ce <= 0;
                    table_entry <= sram_bt_data_out;
                    alloc_start_addr <= sram_bt_data_out[28:16];
                    sram_bm_addr <= sram_bt_data_out[28:21]; // Direct bit select: p_blk[12:5]
                    bitmap_bit_idx <= sram_bt_data_out[20:16]; // Direct bit select: p_blk[4:0]
                    sram_bm_ce <= 1;
                    sram_bm_we <= 0; // Read bitmap
                end
                SCAN_PROCESS: begin
                    sram_bm_ce <= 0;
                    bitmap_entry <= sram_bm_data_out;
                    bitmap_bit <= sram_bm_data_out[bitmap_bit_idx]; // Use pre-computed index
                    if (table_entry[2:0] == 0 && table_entry[15:3] != 0) begin
                        sram_bt_addr <= scan_idx;
                        sram_bt_data_in <= 0;
                        sram_bt_ce <= 1;
                        sram_bt_we <= 1; // Write 0 to block table
                        sram_bm_data_in <= sram_bm_data_out & ~(32'b1 << bitmap_bit_idx); // Clear bit
                        sram_bm_ce <= 1;
                        sram_bm_we <= 1; // Write to bitmap
                    end
                end
                SCAN_UPDATE_BT: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                end
                SCAN_UPDATE_BM: begin
                    sram_bm_ce <= 0;
                    sram_bm_we <= 0;
                    scan_idx <= scan_idx + 1;
                    if (scan_idx == 16383) begin
                        done <= 1;
                    end
                end
            endcase
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            RESET_SRAM: begin
                if (scan_idx == 16383) next_state = IDLE;
            end
            IDLE: next_state = SCAN_READ_BT;
            SCAN_READ_BT: next_state = SCAN_READ_BM;
            SCAN_READ_BM: next_state = SCAN_PROCESS;
            SCAN_PROCESS: begin
                if (table_entry[2:0] == 0 && table_entry[15:3] != 0)
                    next_state = SCAN_UPDATE_BT;
                else
                    next_state = SCAN_UPDATE_BM;
            end
            SCAN_UPDATE_BT: next_state = SCAN_UPDATE_BM;
            SCAN_UPDATE_BM: next_state = (scan_idx == 16383) ? IDLE : SCAN_READ_BT;
            default: next_state = IDLE;
        endcase
    end
endmodule

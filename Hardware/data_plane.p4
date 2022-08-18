 /* -*- P4_16 -*- */


#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "common/headers.p4"
#include "common/util.p4"
/* MACROS */

      
#define CPU_PORT 320

#define ID_VALUE_WIDTH 32
#define S1_SIZE_WIDTH 2
#define S2_SIZE_WIDTH 2
#define S3_SIZE_WIDTH 2
#define S4_SIZE_WIDTH 2
#define S1_SIZE 4
#define S2_SIZE 4
#define S3_SIZE 4
#define S4_SIZE 4
#define S1_COUNTER_VALUE_WIDTH 32
#define S2_COUNTER_VALUE_WIDTH 32
#define S3_COUNTER_VALUE_WIDTH 32
#define S4_COUNTER_VALUE_WIDTH 32
#define THRESHOLD_COUNT 65535
#define PROB_DATA_WIDTH 64


#if __TARGET_TOFINO__ == 1
typedef bit<3> mirror_type_t;
#else
typedef bit<4> mirror_type_t;
#endif
const mirror_type_t MIRROR_TYPE_I2E = 1;
const mirror_type_t MIRROR_TYPE_E2E = 2;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
typedef bit<9>  port_num_t;

struct pair {
    bit<32>     flowID;
    bit<32>     count;
}

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

// resubmit data
header resubmit_h {
    // total 64 bits
    bit<8> d_num;
    bit<8> left_pos;
    bit<8> empty_flag;
    bit<32> min_value;
    bit<8> pad;
}
// mirror
typedef bit<8>  pkt_type_t;
header mirror_h {
  pkt_type_t  pkt_type;
}

struct my_ingress_metadata_t {
    resubmit_h resubmit_data;
    bit<16> srcport;
    bit<16> dstport;
    bit<8> class_flag;
	bit<ID_VALUE_WIDTH> flowID;
    bit<S4_SIZE_WIDTH> mid_idx;
    bit<ID_VALUE_WIDTH> s4_flag;
    bit<ID_VALUE_WIDTH> s3_flag;
    bit<ID_VALUE_WIDTH> s2_flag;

    //stag 1
    bit<S1_SIZE_WIDTH> s1_idx;
    bit<ID_VALUE_WIDTH> s1_d1_flag;
    bit<ID_VALUE_WIDTH> s1_d2_flag;
    bit<ID_VALUE_WIDTH> s1_d3_flag;
    bit<S1_COUNTER_VALUE_WIDTH> s1_d1_count;
    bit<S1_COUNTER_VALUE_WIDTH> s1_d2_count;
    bit<S1_COUNTER_VALUE_WIDTH> s1_d3_count;

    // find min value in s1
    bit<S1_COUNTER_VALUE_WIDTH> cmp_flag1;
    bit<S1_COUNTER_VALUE_WIDTH> cmp_flag2;
    bit<S1_COUNTER_VALUE_WIDTH> cmp_flag3;
    bit<S1_COUNTER_VALUE_WIDTH> s1_min_count;

    bit<32> random_value;
    bit<5> left_pos;

    bit<PROB_DATA_WIDTH> prob_data;
    bit<1> rewrite_flag;
    bit<1> empty_flag;
}

struct my_ingress_headers_t {
    // my change
    ethernet_t  ethernet;
    ipv4_t      ipv4;
    tcp_t       tcp;
    udp_t       udp; 
}

    /***********************  H E A D E R S  ************************/

struct my_egress_headers_t {
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {
}


const bit<16> TYPE_IPV4 = 0x800;
const bit<8> PROTO_TCP = 6;
const bit<8> PROTO_UDP = 17;


/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/
parser IngressParser(packet_in        pkt,
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    //TofinoIngressParser() tofino_parser;
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag){
            1: parse_resubmit;
            0: parse_port_metadata;
        }
    }
    
    state parse_resubmit {
        pkt.extract(meta.resubmit_data);
        transition parse_ethernet;
    }
   state parse_port_metadata {
       pkt.advance(PORT_METADATA_SIZE);
       transition parse_ethernet;
   }
    //
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4   : parse_ipv4;
        }
    }
   
   state parse_ipv4 {
        
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            // PROTO_TCP   : parse_tcp;
            PROTO_UDP   : parse_udp;
             //default: accept;
        }
   }
     
    // state parse_tcp {
    //     pkt.extract(hdr.tcp);
    //     meta.srcport = hdr.tcp.srcPort;
    //     meta.dstport = hdr.tcp.dstPort;
    //     transition accept;
    // }
    
    // state parse_udp {
    //     pkt.extract(hdr.udp);
    //     meta.srcport = hdr.udp.srcPort;
    //     meta.dstport = hdr.udp.dstPort;
    //     transition accept;
    // }

    // test for udp 53
    state parse_udp {
        pkt.extract(hdr.udp);
        meta.srcport = hdr.udp.srcPort;
        meta.dstport = hdr.udp.dstPort;
        transition select(hdr.udp.dstPort) {
            53: accept;
        }
    }
}

   
control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md
     )
{
    // stage 1
    Register<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d1_id;
    Register<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d2_id;
    Register<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d3_id;
    Register<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d1_counter;
    Register<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d2_counter;
    Register<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>>(S1_SIZE)  s1_d3_counter;
    // stage 2
    Register<bit<ID_VALUE_WIDTH>, bit<S2_SIZE_WIDTH>>(S2_SIZE)  s2_id;
    Register<bit<S2_COUNTER_VALUE_WIDTH>, bit<S2_SIZE_WIDTH>>(S2_SIZE)  s2_counter;
    // stage 3
    Register<bit<ID_VALUE_WIDTH>, bit<S3_SIZE_WIDTH>>(S3_SIZE)  s3_id;
    Register<bit<S3_COUNTER_VALUE_WIDTH>, bit<S3_SIZE_WIDTH>>(S3_SIZE)  s3_counter;
    // stage 4
    Register<bit<ID_VALUE_WIDTH>, bit<S4_SIZE_WIDTH>>(S4_SIZE)  s4_id;
    Register<bit<S4_COUNTER_VALUE_WIDTH>, bit<S4_SIZE_WIDTH>>(S4_SIZE)  s4_counter;
	// test pair
	Register<pair, bit<S4_SIZE_WIDTH>>(S4_SIZE) s4_pair;

    
    // insert to stage 4
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S4_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s4_id)
    s4_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };
    RegisterAction<bit<S4_COUNTER_VALUE_WIDTH>, bit<S4_SIZE_WIDTH>, bit<S4_COUNTER_VALUE_WIDTH>>(s4_counter)
    s4_count_update = {
        void apply(inout bit<S4_COUNTER_VALUE_WIDTH> register_data, out bit<S4_COUNTER_VALUE_WIDTH> out_value) {
            if(register_data < THRESHOLD_COUNT) {
                register_data = register_data + 1;
            }
        }
    };

    // insert to stage 3
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S3_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s3_id)
    s3_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };
    RegisterAction<bit<S3_COUNTER_VALUE_WIDTH>, bit<S3_SIZE_WIDTH>, bit<S3_COUNTER_VALUE_WIDTH>>(s3_counter)
    s3_count_update = {
        void apply(inout bit<S3_COUNTER_VALUE_WIDTH> register_data, out bit<S3_COUNTER_VALUE_WIDTH> out_value) {
            if(register_data < THRESHOLD_COUNT) {
                register_data = register_data + 1;
            }
        }
    };

    // insert to stage 2
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S2_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s2_id)
    s2_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };
    RegisterAction<bit<S2_COUNTER_VALUE_WIDTH>, bit<S2_SIZE_WIDTH>, bit<S2_COUNTER_VALUE_WIDTH>>(s2_counter)
    s2_count_update = {
        void apply(inout bit<S2_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            if(register_data < THRESHOLD_COUNT) {
                register_data = register_data + 1;
            }
        }
    };

    // update stage 1
    // for d1
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d1_id)
    s1_d1_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else if(register_data == 0) {
                out_value = 0x2;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };

    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d1_id)
    s1_d1_id_rewrite = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            register_data = meta.flowID;
            
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d1_counter)
    s1_d1_count_update = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = register_data + 1;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d1_counter)
    s1_d1_count_read = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            out_value = register_data;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d1_counter)
    s1_d1_count_rewrite = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = 1;
        }
    };

    // for d2
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d2_id)
    s1_d2_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else if(register_data == 0) {
                out_value = 0x2;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };

    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d2_id)
    s1_d2_id_rewrite = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            register_data = meta.flowID;
            
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d2_counter)
    s1_d2_count_update = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = register_data + 1;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d2_counter)
    s1_d2_count_read = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            out_value = register_data;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d2_counter)
    s1_d2_count_rewrite = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = 1;
        }
    };

    // for d3
    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d3_id)
    s1_d3_id_query = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            if(register_data == meta.flowID)
            {
                out_value = 0x1;
            }
            else if(register_data == 0) {
                out_value = 0x2;
            }
            else
            {
                out_value = 0x0;
            }
            
        }
    };

    RegisterAction<bit<ID_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<ID_VALUE_WIDTH>>(s1_d3_id)
    s1_d3_id_rewrite = {
        void apply(inout bit<ID_VALUE_WIDTH> register_data, out bit<ID_VALUE_WIDTH> out_value) {
            register_data = meta.flowID;
            
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d3_counter)
    s1_d3_count_update = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = register_data + 1;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d3_counter)
    s1_d3_count_read = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            out_value = register_data;
        }
    };

    RegisterAction<bit<S1_COUNTER_VALUE_WIDTH>, bit<S1_SIZE_WIDTH>, bit<S1_COUNTER_VALUE_WIDTH>>(s1_d3_counter)
    s1_d3_count_rewrite = {
        void apply(inout bit<S1_COUNTER_VALUE_WIDTH> register_data, out bit<S2_COUNTER_VALUE_WIDTH> out_value) {
            register_data = 1;
        }
    };
    

    // test register pair
	RegisterAction<pair, bit<S4_SIZE_WIDTH>, bit<S4_COUNTER_VALUE_WIDTH>>(s4_pair)
    s4_pair_insert = {
        void apply(inout pair value, out bit<S4_COUNTER_VALUE_WIDTH> out_value) {
            if(value.flowID == meta.flowID) {
				value.count = value.count + 1;
				out_value = 0x1;
			}else {
				out_value = 0x0;
			}
        }
    };

    /*************************************************************************
    *********************** table action   ***********************************
    *************************************************************************/
    // get flowID
    // Define a custom hash func with CRC polynomial parameters of crc32.
    // crc32 is available in Python
    CRCPolynomial<bit<32>>(32w0x04C11DB7, // polynomial
                           true,          // reversed
                           false,         // use msb?
                           false,         // extended?
                           32w0xFFFFFFFF, // initial shift register value
                           32w0xFFFFFFFF  // result xor
                           ) poly1;
    Hash<bit<ID_VALUE_WIDTH>>(HashAlgorithm_t.CUSTOM, poly1) hash_flowID;
    action ac_get_flowID() {                                  
		meta.flowID= hash_flowID.get({ hdr.ipv4.protocol, 
                                                 hdr.ipv4.srcAddr, 
                                                 hdr.ipv4.dstAddr, 
                                                 meta.srcport, 
                                                 meta.dstport}); 
    }
    @pragma stage 0
    table tb_get_flowID {
        actions = {
            ac_get_flowID;
        }
        default_action = ac_get_flowID;
    }
    
    // get index of stage 2-4
    Hash<bit<S4_SIZE_WIDTH>>(HashAlgorithm_t.CUSTOM, poly1) hash_mid_idx;
    action ac_get_mid_idx() {
        meta.mid_idx = hash_mid_idx.get({ hdr.ipv4.protocol, 
                                                 hdr.ipv4.srcAddr, 
                                                 hdr.ipv4.dstAddr, 
                                                 meta.srcport, 
                                                 meta.dstport}); 
    }
    @pragma stage 0
    table tb_get_mid_idx {
        actions = {
            ac_get_mid_idx;
        }
        default_action = ac_get_mid_idx;
    }

    // get index of stage 1
    Hash<bit<S1_SIZE_WIDTH>>(HashAlgorithm_t.CUSTOM, poly1) hash_s1_idx;
    action ac_get_s1_idx() {
        meta.s1_idx = hash_s1_idx.get({ hdr.ipv4.protocol, 
                                                 hdr.ipv4.srcAddr, 
                                                 hdr.ipv4.dstAddr, 
                                                 meta.srcport, 
                                                 meta.dstport}); 
    }
    @pragma stage 0
    table tb_get_s1_idx {
        actions = {
            ac_get_s1_idx;
        }
        default_action = ac_get_s1_idx;
    }
    
    // insert to s4
    action ac_s4_id_query() {
        meta.s4_flag = s4_id_query.execute(meta.mid_idx);
    }
    table tb_s4_id_query {
        actions = {
            ac_s4_id_query;
        }
        default_action = ac_s4_id_query;
    }

    action ac_s4_count_update() {
        s4_count_update.execute(meta.mid_idx);
    }

    table tb_s4_count_update {
        actions = {
            ac_s4_count_update;
        }
        default_action = ac_s4_count_update;
    }

    // insert to s3
    action ac_s3_id_query() {
        meta.s3_flag = s3_id_query.execute(meta.mid_idx);
    }
    table tb_s3_id_query {
        actions = {
            ac_s3_id_query;
        }
        default_action = ac_s3_id_query;
    }

    action ac_s3_count_update() {
        s3_count_update.execute(meta.mid_idx);
    }

    table tb_s3_count_update {
        actions = {
            ac_s3_count_update;
        }
        default_action = ac_s3_count_update;
    }

    // insert to s2
    action ac_s2_id_query() {
        meta.s2_flag = s2_id_query.execute(meta.mid_idx);
    }
    table tb_s2_id_query {
        actions = {
            ac_s2_id_query;
        }
        default_action = ac_s2_id_query;
    }

    action ac_s2_count_update() {
        s2_count_update.execute(meta.mid_idx);
    }

    table tb_s2_count_update {
        actions = {
            ac_s2_count_update;
        }
        default_action = ac_s2_count_update;
    }
    // try to insert to s1
    // d1
    action ac_s1_d1_id_query() {
        meta.s1_d1_flag = s1_d1_id_query.execute(meta.s1_idx);
    }
    
    table tb_s1_d1_id_query {
        actions = {
            ac_s1_d1_id_query;
        }
        default_action = ac_s1_d1_id_query;
    }

    action ac_s1_d1_count_read() {
        meta.s1_d1_count = s1_d1_count_read.execute(meta.s1_idx);
    }

    action ac_s1_d1_count_update() {
        s1_d1_count_update.execute(meta.s1_idx);
    }
    table tb_s1_d1_count_update {
        key = {
            meta.s1_d1_flag: exact;
        }
        actions = {
            ac_s1_d1_count_read;
            ac_s1_d1_count_update;
        }
        const entries = {
            // 0x0: conflict
            // 0x1: hit
            // 0x2: empty
            (0x1) : ac_s1_d1_count_update();
            (0x0) : ac_s1_d1_count_read();
        }
    }

    // d2
    action ac_s1_d2_id_query() {
        meta.s1_d2_flag = s1_d2_id_query.execute(meta.s1_idx);
    }
    
    table tb_s1_d2_id_query {
        actions = {
            ac_s1_d2_id_query;
        }
        default_action = ac_s1_d2_id_query;
    }

    action ac_s1_d2_count_read() {
        meta.s1_d2_count = s1_d2_count_read.execute(meta.s1_idx);
    }

    action ac_s1_d2_count_update() {
        s1_d2_count_update.execute(meta.s1_idx);
    }
    table tb_s1_d2_count_update {
        key = {
            meta.s1_d2_flag: exact;
        }
        actions = {
            ac_s1_d2_count_read;
            ac_s1_d2_count_update;
        }
        const entries = {
            // 0x0: conflict
            // 0x1: hit
            // 0x2: empty
            (0x1) : ac_s1_d2_count_update();
            (0x0) : ac_s1_d2_count_read();
        }
    }

    // d3
    action ac_s1_d3_id_query() {
        meta.s1_d3_flag = s1_d3_id_query.execute(meta.s1_idx);
    }
    
    table tb_s1_d3_id_query {
        actions = {
            ac_s1_d3_id_query;
        }
        default_action = ac_s1_d3_id_query;
    }

    action ac_s1_d3_count_read() {
        meta.s1_d3_count = s1_d3_count_read.execute(meta.s1_idx);
    }

    action ac_s1_d3_count_update() {
        s1_d3_count_update.execute(meta.s1_idx);
    }
    table tb_s1_d3_count_update {
        key = {
            meta.s1_d3_flag: exact;
        }
        actions = {
            ac_s1_d3_count_read;
            ac_s1_d3_count_update;
        }
        const entries = {
            // 0x0: conflict
            // 0x1: hit
            // 0x2: empty
            (0x1) : ac_s1_d3_count_update();
            (0x0) : ac_s1_d3_count_read();
        }
    }
    
    // check whether exist empty
    action ac_note_s1_d1_empty() {
        // meta.empty_flag = 0x1;
        meta.resubmit_data.empty_flag = 0x1;
        meta.resubmit_data.d_num = 0x1;
    }
    action ac_note_s1_d2_empty() {
        // meta.empty_flag = 0x1;
        meta.resubmit_data.empty_flag = 0x2;
        meta.resubmit_data.d_num = 0x2;
    }
    action ac_note_s1_d3_empty() {
        // meta.empty_flag = 0x1;
        meta.resubmit_data.empty_flag = 0x3;
        meta.resubmit_data.d_num = 0x3;
    }
    table tb_get_s1_empty {
        key = {
            meta.s1_d1_flag: exact;
            meta.s1_d2_flag: exact;
            meta.s1_d3_flag: exact;
        }
        actions = {
            ac_note_s1_d1_empty;
            ac_note_s1_d2_empty;
            ac_note_s1_d3_empty;
        }
        size = 8;
    }

    // find min value in s1
    action ac_s1_compare() {
        meta.cmp_flag1 = meta.s1_d1_count - meta.s1_d2_count;
        meta.cmp_flag2 = meta.s1_d2_count - meta.s1_d3_count;
        meta.cmp_flag3 = meta.s1_d1_count - meta.s1_d3_count;
    }

    table tb_s1_compare {
        actions = {
            ac_s1_compare;
        }
        default_action = ac_s1_compare;
    }

    action ac_set_min_d1() {
        meta.resubmit_data.d_num = 0x1;
        meta.s1_min_count = meta.s1_d1_count;
        meta.resubmit_data.min_value = (bit<32>)meta.s1_d1_count;
    }

    action ac_set_min_d2() {
        meta.resubmit_data.d_num = 0x2;
        meta.s1_min_count = meta.s1_d2_count;
        meta.resubmit_data.min_value = (bit<32>)meta.s1_d2_count;
    }

    action ac_set_min_d3() {
        meta.resubmit_data.d_num = 0x3;
        meta.s1_min_count = meta.s1_d3_count;
        meta.resubmit_data.min_value = (bit<32>)meta.s1_d3_count;
    }

    table tb_set_min_pos {
        key = {
            meta.cmp_flag1: ternary;
            meta.cmp_flag2: ternary;
            meta.cmp_flag3: ternary;
        }
        actions = {
            ac_set_min_d1;
            ac_set_min_d2;
            ac_set_min_d3;
        }
    }

    // resubmit
    action ac_resubmit() {
        ig_dprsr_md.resubmit_type = 2;
    }
    
    table tb_resubmit {
        actions = {
            ac_resubmit;
        }
        default_action = ac_resubmit;
    }

    // rewrite
    // d1
    action ac_rewrite_s1_d1_id() {
        s1_d1_id_rewrite.execute(meta.s1_idx);
    }

    action ac_rewrite_s1_d1_count() {
        s1_d1_count_rewrite.execute(meta.s1_idx);
    }

    @pragma stage 4
    table tb_rewrite_s1_d1_id {
        actions = {
            ac_rewrite_s1_d1_id;
        }
        default_action = ac_rewrite_s1_d1_id;
    }

    @pragma stage 5
    table tb_rewrite_s1_d1_count {
        actions = {
            ac_rewrite_s1_d1_count;
        }
        default_action = ac_rewrite_s1_d1_count;
    }

    // d2
    action ac_rewrite_s1_d2_id() {
        s1_d2_id_rewrite.execute(meta.s1_idx);
    }

    action ac_rewrite_s1_d2_count() {
        s1_d2_count_rewrite.execute(meta.s1_idx);
    }

    @pragma stage 5
    table tb_rewrite_s1_d2_id {
        actions = {
            ac_rewrite_s1_d2_id;
        }
        default_action = ac_rewrite_s1_d2_id;
    }

    @pragma stage 6
    table tb_rewrite_s1_d2_count {
        actions = {
            ac_rewrite_s1_d2_count;
        }
        default_action = ac_rewrite_s1_d2_count;
    }

    // d3
    action ac_rewrite_s1_d3_id() {
        s1_d3_id_rewrite.execute(meta.s1_idx);
    }

    action ac_rewrite_s1_d3_count() {
        s1_d3_count_rewrite.execute(meta.s1_idx);
    }

    @pragma stage 6
    table tb_rewrite_s1_d3_id {
        actions = {
            ac_rewrite_s1_d3_id;
        }
        default_action = ac_rewrite_s1_d3_id;
    }

    @pragma stage 7
    table tb_rewrite_s1_d3_count {
        actions = {
            ac_rewrite_s1_d3_count;
        }
        default_action = ac_rewrite_s1_d3_count;
    }

    // gen random value
    // define 32b random number generator
    Random<bit<32>>() random_func;
    action ac_gen_random() {
        meta.random_value = random_func.get();
    }
    @pragma stage 0
    table tb_gen_random {
        actions = {
            ac_gen_random;
        }
        default_action = ac_gen_random;
    }

    // get position of the most left 1
    action ac_get_left_pos(bit<5> left_pos) {
        meta.left_pos = left_pos;
        meta.resubmit_data.left_pos = (bit<8>) left_pos;
    }

    table tb_get_left_pos {
        key = {
            meta.random_value: ternary;
        }
        actions = {
            ac_get_left_pos;
        }
    }

    // random * min_count
    action ac_assign_value() {
        // meta.prob_data = (bit<PROB_DATA_WIDTH>)meta.resubmit_data.min_value;
        meta.prob_data[31:0] = meta.resubmit_data.min_value;
    }
    @pragma stage 0
    table tb_assign_value {
        actions = {
            ac_assign_value;
        }
        default_action = ac_assign_value;
    }

    action ac_rand_multi_min_0() {
        meta.prob_data[31:0] = meta.resubmit_data.min_value;
    }
    action ac_rand_multi_min_1() {
        meta.prob_data[32:1] = meta.resubmit_data.min_value;
    }

    action ac_rand_multi_min_2() {
        meta.prob_data[33:2] = meta.resubmit_data.min_value;
    }

    action ac_rand_multi_min_3() { meta.prob_data[34:3] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_4() { meta.prob_data[35:4] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_5() { meta.prob_data[36:5] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_6() { meta.prob_data[37:6] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_7() { meta.prob_data[38:7] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_8() { meta.prob_data[39:8] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_9() { meta.prob_data[40:9] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_10() { meta.prob_data[41:10] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_11() { meta.prob_data[42:11] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_12() { meta.prob_data[43:12] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_13() { meta.prob_data[44:13] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_14() { meta.prob_data[45:14] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_15() { meta.prob_data[46:15] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_16() { meta.prob_data[47:16] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_17() { meta.prob_data[48:17] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_18() { meta.prob_data[49:18] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_19() { meta.prob_data[50:19] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_20() { meta.prob_data[51:20] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_21() { meta.prob_data[52:21] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_22() { meta.prob_data[53:22] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_23() { meta.prob_data[54:23] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_24() { meta.prob_data[55:24] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_25() { meta.prob_data[56:25] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_26() { meta.prob_data[57:26] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_27() { meta.prob_data[58:27] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_28() { meta.prob_data[59:28] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_29() { meta.prob_data[60:29] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_30() { meta.prob_data[61:30] = meta.resubmit_data.min_value; }

    action ac_rand_multi_min_31() { meta.prob_data[62:31] = meta.resubmit_data.min_value; }

    @pragma stage 1
    table tb_rand_multi_min {
        key = {
            meta.resubmit_data.left_pos: exact;
        }
        actions = {
            // ac_rand_multi_min_0;
            ac_rand_multi_min_1;
            ac_rand_multi_min_2;
            ac_rand_multi_min_3;
            ac_rand_multi_min_4;
            ac_rand_multi_min_5;
            ac_rand_multi_min_6;
            ac_rand_multi_min_7;
            ac_rand_multi_min_8;
            ac_rand_multi_min_9;
            ac_rand_multi_min_10;
            ac_rand_multi_min_11;
            ac_rand_multi_min_12;
            ac_rand_multi_min_13;
            ac_rand_multi_min_14;
            ac_rand_multi_min_15;
            ac_rand_multi_min_16;
            ac_rand_multi_min_17;
            ac_rand_multi_min_18;
            ac_rand_multi_min_19;
            ac_rand_multi_min_20;
            ac_rand_multi_min_21;
            ac_rand_multi_min_22;
            ac_rand_multi_min_23;
            ac_rand_multi_min_24;
            ac_rand_multi_min_25;
            ac_rand_multi_min_26;
            ac_rand_multi_min_27;
            ac_rand_multi_min_28;
            ac_rand_multi_min_29;
            ac_rand_multi_min_30;
            ac_rand_multi_min_31;
        }
        size = 32;
    }
    action ac_get_rewrite_flag(bit<1> rewrite_flag) {
        // initial:0  prob>2^32: 1 prob<2^32:0
        meta.rewrite_flag = rewrite_flag;
    }
    @pragma stage 2
    table tb_get_rewrite_flag {
        key = {
            meta.prob_data: ternary;
        }
        actions = {
            ac_get_rewrite_flag;
        }
    }

    apply {
        // stage 0
        tb_get_flowID.apply();
        tb_get_mid_idx.apply();
        tb_get_s1_idx.apply();
        tb_gen_random.apply();
        
        
        if (ig_intr_md.resubmit_flag == 0){
            // stage 1
            tb_s4_id_query.apply();
            tb_get_left_pos.apply();
            if(meta.s4_flag == 0x1) {
                // stage 2
                tb_s4_count_update.apply();
            } else {
                tb_s3_id_query.apply();
                if(meta.s3_flag == 0x1) {
                    // stage 3
                    tb_s3_count_update.apply();
                } else {
                    tb_s2_id_query.apply();
                    if(meta.s2_flag == 0x1) {
                        // stage 4
                        tb_s2_count_update.apply();
                    } else {
                        tb_s1_d1_id_query.apply();
                        // stage 5
                        tb_s1_d1_count_update.apply();
                        if(meta.s1_d1_flag != 0x1) {
                            tb_s1_d2_id_query.apply();
                            // stage 6
                            tb_s1_d2_count_update.apply();
                            if(meta.s1_d2_flag != 0x1) {
                                tb_s1_d3_id_query.apply();
                                // stage 7
                                tb_s1_d3_count_update.apply();
                                if(meta.s1_d3_flag != 0x1) {
                                    tb_get_s1_empty.apply();
                                    if(meta.resubmit_data.empty_flag == 0x0){
                                        // stage 8
                                        // find min value
                                        tb_s1_compare.apply();
                                        // stage 9
                                        tb_set_min_pos.apply();
                                    }
                                    
                                    // stage 10
                                    tb_resubmit.apply();
                                    // test
                                    hdr.ipv4.ttl = meta.resubmit_data.d_num;
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            // resubmitted packet
            
            // stage 0
            tb_assign_value.apply();
            // stage 1
            tb_rand_multi_min.apply();
            // stage 2
            tb_get_rewrite_flag.apply();
            if(meta.rewrite_flag == 0x0 || meta.resubmit_data.empty_flag != 0x0)
            {
                if(meta.resubmit_data.d_num == 0x1) {
                    tb_rewrite_s1_d1_id.apply();
                    tb_rewrite_s1_d1_count.apply();
                }

                if(meta.resubmit_data.d_num == 0x2) {
                    tb_rewrite_s1_d2_id.apply();
                    tb_rewrite_s1_d2_count.apply();
                }

                if(meta.resubmit_data.d_num == 0x3) {
                    tb_rewrite_s1_d3_id.apply();
                    tb_rewrite_s1_d3_count.apply();
                }
            }
            
            
        }
        ig_tm_md.ucast_egress_port = 28;
        ig_tm_md.bypass_egress = 1w1;
    }
  
}


control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    Resubmit() resubmit;
    apply {
        // resubmit with resubmit_data
       if (ig_dprsr_md.resubmit_type == 2) {
           resubmit.emit(meta.resubmit_data);
       }
       pkt.emit(hdr);
    }
}



/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

/************ F I N A L   P A C K A G E ******************************/
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EmptyEgressParser(),
    EmptyEgress(),
    EmptyEgressDeparser()
) pipe;

Switch(pipe) main;



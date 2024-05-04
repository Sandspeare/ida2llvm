import idc
import ida_idaapi
import ida_kernwin
import idautils
import ida_name
import ida_bytes
import ida_ida
import ida_funcs
import ida_typeinf
import ida_segment
import ida_nalt
import ida_hexrays
import itertools
import idaapi
import logging
import struct
import numpy as np
import llvmlite.binding as llvm
from llvmlite import ir

from contextlib import suppress

i8ptr = ir.IntType(8).as_pointer()
ptrsize = 64 if ida_idaapi.get_inf_structure().is_64bit() else 32
ptext = {}

def lift_tif(tif: ida_typeinf.tinfo_t, width = -1) -> ir.Type:
    """
    Lifts the given IDA type to corresponding LLVM type.
    If IDA type is an array/struct/tif, type lifting is performed recursively.

    :param tif: the type to lift, in IDA
    :type tif: ida_typeinf.tinfo_t
    :raises NotImplementedError: variadic structs
    :return: lifted LLVM type
    :rtype: ir.Type
    """
    if tif.is_func():
        ida_rettype = tif.get_rettype()                           
        ida_args = (tif.get_nth_arg(i) for i in range(tif.get_nargs()))
        is_vararg = tif.is_vararg_cc()                               
        llvm_rettype = lift_tif(ida_rettype)                            
        llvm_args = (lift_tif(arg) for arg in ida_args)
        return ir.FunctionType(i8ptr if isinstance(llvm_rettype, ir.VoidType) else llvm_rettype, llvm_args, var_arg = is_vararg) 

    elif tif.is_ptr():
        child_tif = tif.get_ptrarr_object()
        if child_tif.is_void():
            return ir.IntType(8).as_pointer()
        return lift_tif(child_tif).as_pointer()

    elif tif.is_array():
        child_tif = tif.get_ptrarr_object()
        element = lift_tif(child_tif)
        count = tif.get_array_nelems()
        if count == 0:
            # an array with an indeterminate number of elements = type pointer
            tif.convert_array_to_ptr()
            return lift_tif(tif)      
        return ir.ArrayType(element, count)

    elif tif.is_void():
        return ir.VoidType()

    elif tif.is_udt():
        udt_data = ida_typeinf.udt_type_data_t()
        tif.get_udt_details(udt_data)
        type_name = tif.get_type_name()
        context = ir.context.global_context
        
        type_name = "struct" if type_name == None else type_name

        if type_name not in context.identified_types:
            struct_t = context.get_identified_type(type_name)
            elementTypes = []
            for idx in range(udt_data.size()):
                udt_member = udt_data.at(idx)
                if udt_member.type.get_type_name() in context.identified_types:
                    elementTypes.append(context.identified_types[udt_member.type.get_type_name()])
                else:
                    element = lift_tif(udt_member.type)
                    elementTypes.append(element)

            struct_t.set_body(*elementTypes)
        return context.get_identified_type(type_name)

    elif tif.is_bool():
        return ir.IntType(1)

    elif tif.is_char():
        return ir.IntType(8)

    elif tif.is_float():
        return ir.FloatType()

    elif tif.is_double():
        return ir.DoubleType()

    # elif tif.is_ldouble():
    #     return ir.DoubleType()

    elif tif.is_decl_int() or tif.is_decl_uint() or tif.is_uint() or tif.is_int():
        return ir.IntType(tif.get_size()*8)
        
    elif tif.is_decl_int16() or tif.is_decl_uint16() or tif.is_uint16() or tif.is_int16():
        return ir.IntType(tif.get_size()*8)
        
    elif tif.is_decl_int32() or tif.is_decl_uint32() or tif.is_uint32() or tif.is_int32():
        return ir.IntType(tif.get_size()*8)
        
    elif tif.is_decl_int64() or tif.is_decl_uint64() or tif.is_uint64() or tif.is_int64():
        return ir.IntType(tif.get_size()*8)
        
    elif tif.is_decl_int128() or tif.is_decl_uint128() or tif.is_uint128() or tif.is_int128():
        return ir.IntType(tif.get_size()*8)

    elif tif.is_ext_arithmetic() or tif.is_arithmetic():
        return ir.IntType(tif.get_size()*8)
        
    else:
        if width != -1:
            return ir.ArrayType(ir.IntType(8), width)
        else:
            return ir.IntType(ptrsize)

def typecast(src: ir.Value, dst_type: ir.Type, builder: ir.IRBuilder, signed: bool = False) -> ir.Value:
    """
    Given some `src`, convert it to type `dst_type`.
    Instructions are emitted into `builder`.

    :param src: value to convert
    :type src: ir.Value
    :param dst_type: destination type
    :type dst_type: ir.Type
    :param builder: builds instructions
    :type builder: ir.IRBuilder
    :param signed: whether to preserve signness, defaults to True
    :type signed: bool, optional
    :raises NotImplementedError: type conversion not supported
    :return: value after typecast
    :rtype: ir.Value   
    """
    if src.type != dst_type:
        if isinstance(src.type, ir.PointerType) and isinstance(dst_type, ir.PointerType):
            return builder.bitcast(src, dst_type)
        elif isinstance(src.type, ir.PointerType) and isinstance(dst_type, ir.IntType):
            return builder.ptrtoint(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.PointerType):
            return builder.inttoptr(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.FloatType):
            return builder.uitofp(src, dst_type)
        elif isinstance(src.type, ir.FloatType) and isinstance(dst_type, ir.IntType):
            if signed == False:
                return builder.fptoui(src, dst_type)
            else:
                return builder.fptosi(src, dst_type)
        elif isinstance(src.type, ir.DoubleType) and isinstance(dst_type, ir.IntType):
            if signed == False:
                return builder.fptoui(src, dst_type)
            else:
                return builder.fptosi(src, dst_type)
        elif isinstance(src.type, ir.FloatType) and isinstance(dst_type, ir.FloatType):
            return src
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.IntType) and src.type.width < dst_type.width:
            if signed:
                return builder.sext(src, dst_type)
            else:
                return builder.zext(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.IntType) and src.type.width > dst_type.width:
            return builder.trunc(src, dst_type)

        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.DoubleType):
            return builder.uitofp(src, dst_type)

        elif isinstance(src.type, ir.FloatType) and isinstance(dst_type, ir.DoubleType):
            return builder.fpext(src, dst_type)

        elif isinstance(src.type, ir.DoubleType) and isinstance(dst_type, ir.FloatType):
            return builder.fptrunc(src, dst_type)

        elif (isinstance(src.type, ir.DoubleType) or isinstance(src.type, ir.FloatType)) and isinstance(dst_type, ir.PointerType):
            if signed == False:
                tmp =  builder.fptoui(src, ir.IntType(ptrsize))
            else:
                tmp = builder.fptosi(src, ir.IntType(ptrsize))
            return builder.inttoptr(tmp, dst_type)

        elif (isinstance(dst_type, ir.DoubleType) or isinstance(dst_type, ir.FloatType)) and isinstance(src.type, ir.PointerType):
            tmp = builder.ptrtoint(src, ir.IntType(ptrsize))
            return builder.uitofp(tmp, dst_type)

        elif isinstance(dst_type, ir.IdentifiedStructType) or isinstance(dst_type, ir.ArrayType): 
            with builder.goto_entry_block():
                tmp = builder.alloca(src.type)
            builder.store(src, tmp)
            src = builder.load(builder.bitcast(tmp, dst_type.as_pointer()))

        elif isinstance(src.type, ir.IdentifiedStructType) or isinstance(src.type, ir.ArrayType): 
            with builder.goto_entry_block():
                tmp = builder.alloca(src.type)
            builder.store(src, tmp)
            src = builder.load(builder.bitcast(tmp, dst_type.as_pointer()))

        else:
            return builder.bitcast(src, dst_type)
    return src

def storecast(src, dst, builder):
    """
    This function cast type of dst into pointer of src.
    """
    if dst != None and dst.type != src.type.as_pointer():
        dst = typecast(dst, src.type.as_pointer(), builder) 
    return dst

def get_offset_to(builder: ir.IRBuilder, arg: ir.Value, off: int = 0) -> ir.Value:
    """
    A Value can be indexed relative to some offset.

    :param arg: value to index from
    :type arg: ir.Value
    :param off: offset to index, defaults to 0
    :type off: int, optional
    :return: value after indexing by off
    :rtype: ir.Value
    """
    if isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.ArrayType):
        arr = arg.type.pointee
        td = llvm.create_target_data("e")
        size = arr.element.get_abi_size(td)
        return builder.gep(arg, (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), off // size),))
    # elif isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.LiteralStructType):
    #     return typecast(arg, ir.IntType(8).as_pointer(), builder)
    elif isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.IdentifiedStructType):
        return typecast(arg, ir.IntType(8).as_pointer(), builder)
    elif isinstance(arg.type, ir.PointerType) and off > 0:
        td = llvm.create_target_data("e")
        size = arg.type.pointee.get_abi_size(td)
        return builder.gep(arg, (ir.Constant(ir.IntType(32), off // size),))
    else:
        return arg

def dedereference(arg: ir.Value) -> ir.Value:
    """
    A memory address is deferenced if the memory at the address is loaded.
    In LLVM, a LoadInstruction instructs the CPU to perform the dereferencing.

    In cases where we wish to retrieve the memory address, we "de-dereference".
    - this is needed as IDA microcode treats all LVARS as registers
    - whereas during lifting we treat all LVARS as stack variables (in accordance to LLVM SSA)

    :param arg: value to de-dereference
    :type arg: ir.Value
    :raises NotImplementedError: arg is not of type LoadInstr
    :return: original memory address
    :rtype: ir.Value
    """
    if isinstance(arg, ir.LoadInstr):
        return arg.operands[0]
    elif isinstance(arg.type, ir.PointerType):
        return arg
    else:
        raise NotImplementedError(f"not implemented: get reference for object {arg} of type {arg.type}")


def lift_type_from_address(ea: int, pfunc=None):
    if ida_funcs.get_func(ea) != None and ida_segment.segtype(ea) & ida_segment.SEG_XTRN:
        # let's assume its a function that returns ONE register and takes in variadic arguments
        ida_func_details = ida_typeinf.func_type_data_t()
        void = ida_typeinf.tinfo_t()
        void.create_simple_type(ida_typeinf.BTF_VOID)
        ida_func_details.rettype = void
        ida_func_details.cc = ida_typeinf.CM_CC_ELLIPSIS | ida_typeinf.CC_CDECL_OK

        function_tinfo = ida_typeinf.tinfo_t()
        function_tinfo.create_func(ida_func_details)
        return function_tinfo

    if ea in ptext:
        return ptext[ea].type
            
    tif = ida_typeinf.tinfo_t()
    ida_nalt.get_tinfo(tif, ea)
    if not ida_nalt.get_tinfo(tif, ea):
        ida_typeinf.guess_tinfo(tif, ea)
    return tif

def analyze_insn(module, ida_insn, ea):
    """
    This function analyze is there mismatch function call.
    A call B with 3 arguments but B has 4 arguments.
    This typically owing to IDA type propagation, sometimes will be solved by re-decompile.
    """
    if ida_insn.opcode == ida_hexrays.m_call:
        callnum = len(ida_insn.d.f.args)
        if ida_insn.l.t == ida_hexrays.mop_v: 
            temp_ea = ida_insn.l.g
            func_name = ida_name.get_name(temp_ea)
            if ((ida_funcs.get_func(temp_ea) is not None)
            and (ida_funcs.get_func(temp_ea).flags & ida_funcs.FUNC_THUNK)): 
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(ida_funcs.get_func(temp_ea))
                if tfunc_ea != ida_idaapi.BADADDR:
                    temp_ea = tfunc_ea
                    func_name = ida_name.get_name(temp_ea)

            tif = lift_type_from_address(temp_ea)
            if tif.is_func() or tif.is_funcptr():
                argnum = tif.get_nargs()
                with suppress(KeyError):
                    if func_name == "":
                        func_name = f"data_{hex(ea)[2:]}"
                    if hasattr(module.get_global(func_name
                                                 ), "args"):
                        argnum = len(module.get_global(func_name).args)

                if callnum != argnum:
                    ida_hf = ida_hexrays.hexrays_failure_t()
                    try:
                        pfunc = ida_hexrays.decompile(temp_ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE)
                        if pfunc != None:
                            ptext[temp_ea] = pfunc
                    except:
                        pass

                    try:
                        pfunc = ida_hexrays.decompile(ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE) 
                        if pfunc != None:
                            ptext[ea] = pfunc
                    except:
                        return

    if ida_insn.l.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.l.d, ea)
    if ida_insn.r.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.r.d, ea)
    if ida_insn.d.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.d.d, ea)
    return

def lift_from_address(module: ir.Module, ea: int, typ: ir.Type = None):
    if typ is None:
        tif = lift_type_from_address(ea)
        typ = lift_tif(tif)
    return _lift_from_address(module, ea, typ) 

def _lift_from_address(module: ir.Module, ea: int, typ: ir.Type):
    if isinstance(typ, ir.FunctionType):
        func_name = ida_name.get_name(ea)
        if func_name == "":
            func_name = f"data_{hex(ea)[2:]}"
        res = module.get_global(func_name)
        res.lvars = dict() 
        if ea in ptext:
            pfunc = ptext[ea]
        else:
            return res

        # refresh function call.
        # some functions caller and signature maybe mismatch. refresh with re-decompile.
        mba = pfunc.mba
        for index in range(mba.qty):
            ida_blk = mba.get_mblock(index)
            ida_insn = ida_blk.head
            while ida_insn is not None:
                analyze_insn(module, ida_insn, ea)
                ida_insn = ida_insn.next

        if ea in ptext:
            pfunc = ptext[ea]
        else:
            return res
        
        mba = pfunc.mba
        for index in range(mba.qty):
            res.append_basic_block(name = f"@{index}")

        ida_func_details = ida_typeinf.func_type_data_t()
        tif = lift_type_from_address(ea, pfunc) 
        tif.get_func_details(ida_func_details)
        names = [] 

        builder = ir.IRBuilder(res.entry_basic_block)

        with builder.goto_entry_block():
            # declare function results as stack variable
            if not isinstance(typ.return_type, ir.VoidType):
                res.lvars["funcresult"] = builder.alloca(typ.return_type, name = "funcresult")

            for lvar in list(pfunc.lvars): 
                if lvar.is_result_var:
                    continue
                arg_t = lift_tif(lvar.tif)
                res.lvars[lvar.name] = builder.alloca(arg_t, name = lvar.name)
                if lvar.is_arg_var:
                    names.append(lvar.name)

            # if function is variadic, declare va_start intrinsic
            if tif.is_vararg_cc() and typ.var_arg:  
                ptr = builder.alloca(ir.IntType(8).as_pointer(), name = "ArgList")
                res.lvars["ArgList"] = ptr
                va_start = module.declare_intrinsic('llvm.va_start', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
                ptr = builder.load(ptr)
                builder.call(va_start, (ptr, ))

            # store stack variables
            for arg, arg_n in zip(res.args, names):   
                arg = typecast(arg, res.lvars[arg_n].type.pointee, builder) 
                builder.store(arg, res.lvars[arg_n]) 

        with builder.goto_block(res.blocks[-1]):
            if isinstance(typ.return_type, ir.VoidType):
                builder.ret_void() 
            else:
                builder.ret(builder.load(res.lvars["funcresult"])) 

        # lift each bblk in cfg
        for index, blk in enumerate(res.blocks):
            ida_blk = mba.get_mblock(index)
            ida_insn = ida_blk.head
            while ida_insn is not None:
                lift_insn(ida_insn, blk, builder)
                ida_insn = ida_insn.next

            if not blk.is_terminated and index + 1 < len(res.blocks):
                with builder.goto_block(blk):
                    builder.branch(res.blocks[index + 1])

        # if function is variadic, declare va_end intrinsic
        if tif.is_vararg_cc() and typ.var_arg:
            ptr = res.lvars["ArgList"]
            va_end = module.declare_intrinsic('llvm.va_end', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
            with builder.goto_block(res.blocks[-1]):
                ptr = builder.load(ptr)
                builder.call(va_end, (ptr, ))
        return res

    elif isinstance(typ, ir.IntType):
        # should probably check endianness, BOOL type is IntType(1)
        r = ida_bytes.get_bytes(ea, 1 if typ.width // 8 < 1 else typ.width // 8)
        return typ(0) if r == None else typ(int.from_bytes(r, "little"))
    elif isinstance(typ, ir.FloatType):
        r = ida_bytes.get_bytes(ea, 4)
        value = struct.unpack('f', r)
        return typ(np.float32(0)) if r == None else typ(np.float32(value[0]))
    elif isinstance(typ, ir.DoubleType):
        r = ida_bytes.get_bytes(ea, 8)
        value = struct.unpack('d', r)
        return typ(np.float64(0)) if r == None else typ(np.float64(value[0]))
    elif isinstance(typ, ir.PointerType):
        val = ir.Constant(typ, None)
        return val
    elif isinstance(typ, ir.ArrayType):
        td = llvm.create_target_data("e")
        subSize = typ.element.get_abi_size(td)
        array = [ lift_from_address(module, sub_ea, typ.element)
            for sub_ea in range(ea, ea + subSize * typ.count, subSize)
        ]
        return ir.Constant.literal_array(array)
    elif isinstance(typ, ir.LiteralStructType) or isinstance(typ, ir.IdentifiedStructType):
        td = llvm.create_target_data("e")
        structEles = []
        for el in typ.elements:
            structEle = lift_from_address(module, ea, el)
            structEles.append(structEle)
            subSize = el.get_abi_size(td)
            ea += subSize
        return ir.Constant(typ, structEles)
    else:
        raise NotImplementedError(f"object at {hex(ea)} is of unsupported type {typ}")

def str2size(str_size: str):
    """
    Converts a string representing memory size into its size in bits. 

    :param str_size: string describing size
    :type str_size: str
    :return: size of string, in bits
    :rtype: int
    """
    if str_size == "byte":
        return 8
    elif str_size == "word":
        return 16
    elif str_size == "dword":
        return 32
    elif str_size == "qword":
        return 64
    else:
        raise AssertionError("string size must be one of byte/word/dword/qword")

def lift_intrinsic_function(module: ir.Module, func_name: str):
    """
    Lifts IDA macros to corresponding LLVM intrinsics.

    Hexray's decompiler recognises higher-level functions at the Microcode level.
        Such ida_hexrays:mop_t objects are typed as ida_hexrays.mop_h (auxillary function member)
        
        This improves decompiler output, representing operations that cannot be mapped to nice C code
        (https://hex-rays.com/blog/igors-tip-of-the-week-67-decompiler-helpers/).

        For relevant #define macros, refer to IDA SDK: `defs.h` and `pro.h`.

    LLVM intrinsics have well known names and semantics and are required to follow certain restrictions.

    :param module: _description_
    :type module: ir.Module
    :param func_name: _description_
    :type func_name: str
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: _type_
    """
    # retrieve intrinsic function if it already exists
    with suppress(KeyError):
        return module.get_global(func_name)

    if func_name == "sadd_overflow":
        typ = ir.LiteralStructType((ir.IntType(64), ir.IntType(1)))
        return  module.declare_intrinsic('sadd_overflow', fnty=ir.FunctionType(typ.as_pointer(), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "__OFSUB__":
        return  module.declare_intrinsic('__OFSUB__', fnty=ir.FunctionType(ir.IntType(1), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "_mm_cvtsi128_si32":
        return  module.declare_intrinsic('_mm_cvtsi128_si32', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(128)]))

    elif func_name == "_BitScanReverse":
        return  module.declare_intrinsic('_BitScanReverse', fnty=ir.FunctionType(i8ptr, [ir.IntType(32), ir.IntType(32)]))

    elif func_name == "__FYL2X__":
        return  module.declare_intrinsic('__FYL2X__', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]))

    elif func_name == "__FYL2P__":
        return  module.declare_intrinsic('__FYL2P__', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]))
       
    elif func_name == "fabs":
        return  module.declare_intrinsic('fabs', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]))

    elif func_name == "fabsf":
        return  module.declare_intrinsic('fabsf', fnty=ir.FunctionType(ir.FloatType(), [ir.FloatType()]))

    elif func_name == "fabsl":
        return  module.declare_intrinsic('fabs', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]))

    elif func_name == "memcpy":
        return  module.declare_intrinsic('memcpy', fnty=ir.FunctionType(i8ptr, [i8ptr, i8ptr, ir.IntType(64)]))

    elif func_name == "_byteswap_ulong":
        return  module.declare_intrinsic('_byteswap_ulong', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(32)]))

    elif func_name == "_byteswap_uint64":
        return  module.declare_intrinsic('_byteswap_uint64', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)]))

    elif func_name == "memset":
        return  module.declare_intrinsic('memset', fnty=ir.FunctionType(i8ptr, [i8ptr, ir.IntType(32), ir.IntType(32)]))

    elif func_name == "abs64":
        return  module.declare_intrinsic('abs64', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)]))

    elif func_name == "__PAIR64__":
        return  module.declare_intrinsic('__PAIR64__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(32), ir.IntType(32)]))

    elif func_name == "__PAIR128__":
        return  module.declare_intrinsic('__PAIR128__', fnty=ir.FunctionType(ir.IntType(128), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "__PAIR32__":
        return  module.declare_intrinsic('__PAIR32__', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(16), ir.IntType(16)]))

    elif func_name == "__PAIR16__":
        return  module.declare_intrinsic('__PAIR16__', fnty=ir.FunctionType(ir.IntType(16), [ir.IntType(8), ir.IntType(8)]))

    elif func_name == "_BitScanReverse64":
        return  module.declare_intrinsic('_BitScanReverse64', fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]))

    elif func_name == "_BitScanForward64":
        return  module.declare_intrinsic('_BitScanForward64', fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]))


    elif func_name == "__halt":
        fty = ir.FunctionType(ir.VoidType(), [])
        f = ir.Function(module, fty, "__halt")
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        builder.asm(fty, "hlt", "", (), True)
        builder.ret_void()
        return f

    elif func_name == "is_mul_ok":
        is_mul_ok = module.declare_intrinsic('is_mul_ok', fnty=ir.FunctionType(ir.IntType(8), [ir.IntType(64), ir.IntType(64)]))
        return is_mul_ok

    elif func_name == "va_start":
        va_start = module.declare_intrinsic('va_start', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
        return va_start

    elif func_name == "va_arg":
        va_arg = module.declare_intrinsic('va_arg', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer()]))
        return va_arg

    elif func_name == "va_end":
        va_arg = module.declare_intrinsic('va_end', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
        return va_arg

    elif func_name == "_QWORD":
        fQWORD = module.declare_intrinsic('IDA_QWORD', fnty=ir.FunctionType(ir.IntType(8).as_pointer(), []))
        return fQWORD

    elif func_name == "__ROL8__":
        f = module.declare_intrinsic('__ROL8__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f

    elif func_name == "__ROL4__":
        f = module.declare_intrinsic('__ROL4__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f
        
    elif func_name == "__ROR4__":
        f = module.declare_intrinsic('__ROR4__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f

    elif func_name == "__ROR8__":
        f = module.declare_intrinsic('__ROR8__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f
        
    elif func_name.startswith("__readfs"):
        _, size = func_name.split("__readfs")
        size = str2size(size)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), 65536)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.storage_class = "thread_local"
            fs_reg.initializer = fs_reg_typ(None)
        try:
            threadlocal_f = module.get_global('llvm.threadlocal.address')
        except KeyError:
            f_argty = (i8ptr, )
            fnty = ir.FunctionType(i8ptr, f_argty)
            threadlocal_f = module.declare_intrinsic('llvm.threadlocal.address', f_argty, fnty)

        fty = ir.FunctionType(ir.IntType(size), [ir.IntType(32),])

        f = ir.Function(module, fty, func_name)
        offset, = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        fs_reg = typecast(fs_reg, ir.IntType(8).as_pointer(), builder)
        threadlocal_address = builder.call(threadlocal_f, (fs_reg, ))
        pointer = builder.gep(threadlocal_address, (offset,))
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        res = builder.load(pointer)
        builder.ret(res)

        return f

    elif func_name.startswith("__writefs"):
        _, size = func_name.split("__writefs")
        size = str2size(size)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), 65536)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.storage_class = "thread_local"
            fs_reg.initializer = fs_reg_typ(None)            
        try:
            threadlocal_f = module.get_global('llvm.threadlocal.address')
        except KeyError:
            f_argty = (i8ptr, )
            fnty = ir.FunctionType(i8ptr, f_argty)
            threadlocal_f = module.declare_intrinsic('llvm.threadlocal.address', f_argty, fnty)

        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32), ir.IntType(size)])

        f = ir.Function(module, fty, func_name)
        offset, value  = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        fs_reg = typecast(fs_reg, ir.IntType(8).as_pointer(), builder)
        threadlocal_address = builder.call(threadlocal_f, (fs_reg, ))
        pointer = builder.gep(threadlocal_address, (offset,))
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        builder.store(value, pointer)
        builder.ret_void()

        return f

    elif func_name.startswith("sys_"):
        fty = ir.FunctionType(ir.IntType(64), [], var_arg=True)
        f = ir.Function(module, fty, func_name)
        return f

    elif func_name.startswith("_InterlockedCompareExchange") or func_name.startswith("_InterlockedExchange"):
        fty = ir.FunctionType(ir.IntType(64), [], var_arg=True)
        f = ir.Function(module, fty, func_name)
        return f

    else:
        raise NotImplementedError(f"NotImplementedError {func_name}")  

def lift_function(module: ir.Module, func_name: str, is_declare: bool, ea = None, tif: ida_typeinf.tinfo_t = None):
    """
    Declares function given its name. 
    If `is_declare` is False, also define the function by recursively.
    If `tif` is given, enforce function type as given.
    lifting its instructions in IDA decompiler output.
    Heavylifting is done in `lift_from_address`.

    :param module: parent module of function
    :type module: ir.Module
    :param func_name: name of function to lift
    :type func_name: str
    :param is_declare: is the function declare only?
    :type is_declare: bool
    :param tif: function type, defaults to None
    :type tif: ida_typeinf.tinfo_t, optional
    :return: lifted function
    :rtype: ir.Function
    """
    if func_name == "":
        func_name = f"data_{hex(ea)[2:]}"
    with suppress(NotImplementedError):
        return lift_intrinsic_function(module, func_name)

    with suppress(KeyError):
        return module.get_global(func_name) 

    func_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, func_name)
    if ida_segment.segtype(func_ea) & ida_segment.SEG_XTRN:
        is_declare = True 

    if func_ea == ida_idaapi.BADADDR:
        func_ea = ea

    assert func_ea != ida_idaapi.BADADDR
    if tif is None:
        tif = lift_type_from_address(func_ea) 

    typ = lift_tif(tif) 
    if isinstance(typ, ir.PointerType):
        print()
    res = ir.Function(module, typ, func_name)
    if is_declare:
        return res
    return lift_from_address(module, func_ea, typ) 

def calc_instsize(typ):
    """
    This function calculate inst width
    """
    if isinstance(typ, ir.PointerType):
        return ptrsize
    elif isinstance(typ, ir.ArrayType):
        return -1
    elif isinstance(typ, ir.IdentifiedStructType):
        return -1
    elif isinstance(typ, ir.FloatType):
        return 32
    elif isinstance(typ, ir.DoubleType):
        return 64
    else:
        return typ.width

def lift_mop(mop: ida_hexrays.mop_t, blk: ir.Block, builder: ir.IRBuilder, dest = False, knowntyp = None) -> ir.Value:
    """
    This function lift mop of microcode into llvm.
    """
    builder.position_at_end(blk)
    if mop.t == ida_hexrays.mop_r: # register value
        return None
    elif mop.t == ida_hexrays.mop_n: # immediate value
        res = ir.Constant(ir.IntType(mop.size * 8), mop.nnn.value)
        res.parent = blk
        return res
    elif mop.t == ida_hexrays.mop_d: # another instruction
        d = lift_insn(mop.d, blk, builder)
        if isinstance(d.type, ir.VoidType):
            pass
        elif mop.size == -1:
            pass
        elif isinstance(mop, ida_hexrays.mcallarg_t):
            lltype = lift_tif(mop.type)
            d = typecast(d, lltype, builder, signed=mop.type.is_signed())  
        elif knowntyp != None:
            d = typecast(d, knowntyp, builder)
        elif calc_instsize(d.type) != mop.size * 8:
            d = typecast(d, ir.IntType(mop.size * 8), builder)
        return d
    elif mop.t == ida_hexrays.mop_l: # local variables
        lvar = mop.l.var()
        name = "funcresult" if lvar.is_result_var else lvar.name
        off = mop.l.off
        func = blk.parent
        llvm_arg = func.lvars[name]
        llvm_arg = get_offset_to(builder, llvm_arg, off)
        if mop.size == -1:
            pass
        elif knowntyp != None:
            llvm_arg = typecast(llvm_arg, knowntyp, builder)
        elif calc_instsize(llvm_arg.type.pointee) != mop.size * 8:
            llvm_arg = typecast(llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder)
        return llvm_arg if dest else builder.load(llvm_arg)
    elif mop.t == ida_hexrays.mop_S: # stack variables
        name = "stack"
        func = blk.parent
        if name not in func.lvars:
            with builder.goto_entry_block():                
                func.lvars[name] = builder.alloca(ir.IntType(ptrsize), name = name)
        llvm_arg = func.lvars[name]
        llvm_arg = get_offset_to(builder, llvm_arg, mop.s.off)
        if mop.size == -1:
            pass
        elif knowntyp != None:
            d = typecast(d, knowntyp, builder)
        elif calc_instsize(llvm_arg.type.pointee) != mop.size * 8:
            llvm_arg = typecast(llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder)
        return llvm_arg if dest else builder.load(llvm_arg) 
        if (hasattr(llvm_arg.type.pointee, "width") and llvm_arg.type.pointee.width != mop.size * 8) and mop.size != -1:
            llvm_arg = typecast(llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder)
        return llvm_arg if dest else builder.load(llvm_arg) 
    elif mop.t == ida_hexrays.mop_b: # block number (used in jmp\call instruction)
        return blk.parent.blocks[mop.b]
    elif mop.t == ida_hexrays.mop_v: # global variable
        ea = mop.g
        name = ida_name.get_name(ea)
        if name == "":
            name = f"data_{hex(ea)[2:]}"

        tif = lift_type_from_address(ea)
        if tif.is_func() or tif.is_funcptr():
            with suppress(KeyError):
                g = blk.parent.parent.get_global(name)
                return g
            if tif.is_funcptr():
                tif = tif.get_ptrarr_object()
            # if function is a thunk function, define the actual function instead
            if ((ida_funcs.get_func(ea) is not None)
            and (ida_funcs.get_func(ea).flags & ida_funcs.FUNC_THUNK)): 
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(ida_funcs.get_func(ea))
                if tfunc_ea != ida_idaapi.BADADDR:
                    ea = tfunc_ea
                    name = ida_name.get_name(ea)
                    if name == "":
                        name = f"data_{hex(ea)[2:]}"
                    tif = lift_type_from_address(ea)
            # if no function definition,
            if ((ida_funcs.get_func(ea) is None)
            # or if the function is a library function,
            or (ida_funcs.get_func(ea).flags & ida_funcs.FUNC_LIB) 
            # or if the function is declared in a XTRN segment,
            or ida_segment.segtype(ea) & ida_segment.SEG_XTRN): 
                # return function declaration
                g = lift_function(blk.parent.parent, name, True, ea, tif)
            else:
                g = lift_function(blk.parent.parent, name, False, ea, tif)
            return g
                            
        else:  
            if name in blk.parent.parent.globals:
                g = blk.parent.parent.get_global(name)
            else:
                tif = lift_type_from_address(ea)
                typ = lift_tif(tif)
                g_cmt = lift_from_address(blk.parent.parent, ea, typ)
                g = ir.GlobalVariable(blk.parent.parent, g_cmt.type, name = name)
                g.initializer = g_cmt

            if isinstance(g.type.pointee, ir.IdentifiedStructType) or isinstance(g.type.pointee, ir.ArrayType):
                g = builder.gep(g, (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)))
            if mop.size == -1:
                pass
            elif knowntyp != None:
                g = typecast(g, knowntyp, builder)
            elif calc_instsize(g.type.pointee) != mop.size * 8:
                g = typecast(g, ir.IntType(mop.size * 8).as_pointer(), builder)     
            return g if dest else builder.load(g)
    elif mop.t == ida_hexrays.mop_f: # function call information
        mcallinfo = mop.f
        f_args = []
        f_ret = []
        for i in range(mcallinfo.retregs.size()):
            mopt = mcallinfo.retregs.at(i)
            f_ret.append(lift_mop(mopt, blk, builder, dest))
        for arg in mcallinfo.args:
            typ = lift_tif(arg.type)
            f_arg = lift_mop(arg, blk, builder, dest, typ.as_pointer())

            if arg.t == ida_hexrays.mop_h and f_arg == None:
                f_arg = blk.parent.parent.declare_intrinsic(arg.helper, fnty=ir.FunctionType(typ, []))

            if arg.t == ida_hexrays.mop_r and f_arg == None:
                name = "fs"
                func = blk.parent
                if name not in func.lvars:
                    with builder.goto_entry_block():                
                        func.lvars[name] = builder.alloca(ir.IntType(16), name = name)
                llvm_arg = func.lvars[name]
                f_arg = llvm_arg if mop.size == -1 else builder.load(llvm_arg)

            f_arg = typecast(f_arg, typ, builder)
            f_args.append(f_arg)
        return f_ret, f_args
    elif mop.t == ida_hexrays.mop_a: # operating number address (mop_l\mop_v\mop_S\mop_r)
        mop_addr = mop.a
        val = lift_mop(mop_addr, blk, builder, True) 
        if isinstance(mop, ida_hexrays.mcallarg_t):
            lltype = lift_tif(mop.type)
            val = typecast(val, lltype, builder)
        elif isinstance(mop, ida_hexrays.mop_addr_t):
            lltype = lift_tif(mop.type)
            val = typecast(val, lltype, builder)
        elif knowntyp != None:
            val = typecast(val, knowntyp, builder)
        return val
    elif mop.t == ida_hexrays.mop_h: # auxiliary function number
        with suppress(NotImplementedError):
            return lift_intrinsic_function(blk.parent.parent, mop.helper)
        return None
    elif mop.t == ida_hexrays.mop_str: # string constant
        str_csnt = mop.cstr
        strType = ir.ArrayType(ir.IntType(8), len(str_csnt))
        g = ir.GlobalVariable(blk.parent.parent, strType, name=f"cstr_{len(blk.parent.parent.globals)}")
        g.initializer = ir.Constant(strType, bytearray(str_csnt.encode("utf-8")))
        g.linkage = "private"
        g.global_constant = True
        return typecast(g, ir.IntType(8).as_pointer(), builder)
    elif mop.t == ida_hexrays.mop_c: # switch case and target
        mcases = {}
        for i in range(mop.c.size()):
            dst = mop.c.targets[i]  
            if mop.c.values[i].size() == 0:
                mcases["default"] = dst 
            for j in range(mop.c.values[i].size()):
                src = mop.c.values[i][j]
                mcases[src] = dst
        return mcases
    elif mop.t == ida_hexrays.mop_fn:
        # IDA get float value may be crash in some cases
        try:
            fp = mop.fpc.fnum.float
        except:
            fp = 1.0
        if mop.size == 4:
            typ = ir.FloatType()
        elif mop.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType()
        return ir.Constant(typ, fp)         
    elif mop.t == ida_hexrays.mop_p:
        f = lift_intrinsic_function(blk.parent.parent, f"__PAIR{mop.size*8}__")
        l = lift_mop(mop.pair.hop, blk, builder, dest)
        r = lift_mop(mop.pair.lop, blk, builder, dest)
        l = typecast(l, ir.IntType(mop.size*4), builder)
        r = typecast(r, ir.IntType(mop.size*4), builder)
        return builder.call(f, (l, r))
    elif mop.t == ida_hexrays.mop_sc:
        pass
    elif mop.t == ida_hexrays.mop_z:
        return None
    mop_descs = {ida_hexrays.mop_r: "register value",
                ida_hexrays.mop_n: "immediate value",
                ida_hexrays.mop_d: "another instruction",
                ida_hexrays.mop_l: "local variables",
                ida_hexrays.mop_S: "stack variables",
                ida_hexrays.mop_b: "block number (used in jmp\call instruction)",
                ida_hexrays.mop_v: "global variable",
                ida_hexrays.mop_f: "function call information",
                ida_hexrays.mop_a: "operating number address (mop_l\mop_v\mop_S\mop_r)",
                ida_hexrays.mop_h: "auxiliary function number",
                ida_hexrays.mop_str: "string constant",
                ida_hexrays.mop_c: "switch case and target",
                ida_hexrays.mop_fn: "floating points constant",
                ida_hexrays.mop_p: "the number of operations is correct",
                ida_hexrays.mop_sc: "decentralized operation information"
    }
    raise NotImplementedError(f"not implemented: {mop.dstr()} of type {mop_descs[mop.t]}")

def _store_as(l: ir.Value, d: ir.Value, blk: ir.Block, builder: ir.IRBuilder, d_typ: ir.Type = None, signed: bool = True):
    """
    Private helper function to store value to destination.
    """
    if d is None:  # destination does not exist
        return l

    d = dedereference(d)
    if d_typ:
        d = typecast(d, d_typ, builder, signed)
    assert isinstance(d.type, ir.PointerType)

    if isinstance(d.type.pointee, ir.ArrayType):
        arrtoptr = d.type.pointee.element.as_pointer()
        d = typecast(d, arrtoptr.as_pointer(), builder, signed)

    if isinstance(l.type, ir.VoidType):
        return

    with suppress(AttributeError):
        if isinstance(l.type.pointee, ir.IdentifiedStructType) or isinstance(l.type.pointee, ir.ArrayType):
            dest, src = d, l
            td = llvm.create_target_data("e")
            length = ir.Constant(ir.IntType(64), l.type.pointee.get_abi_size(td))
            memcpy = lift_intrinsic_function(blk.parent.parent, "memcpy")
            src = typecast(src, ir.IntType(8).as_pointer(), builder)
            dest = typecast(dest, ir.IntType(8).as_pointer(), builder)
            return builder.call(memcpy, (dest, src, length))
    
    if isinstance(d.type.pointee, ir.IdentifiedStructType):
        d = typecast(d, l.type.as_pointer(), builder)
    else:
        l = typecast(l, d.type.pointee, builder, signed)

    return builder.store(l, d)

def create_intrinsic_function(module: ir.Module, func_name: str, ftif):
    """
    This function create intrinsic function for IDA helper function.
    """
    argtypes = []
    for arg in ftif.args:
        argtypes.append(lift_tif(arg.type))

    rettype = lift_tif(ftif.return_type)
    if isinstance(rettype, ir.VoidType):
        rettype = i8ptr
    return module.declare_intrinsic(func_name, fnty=ir.FunctionType(rettype, argtypes))

def float_type(size):
    """
    This function return point type for specific size.
    error: llvmlite do not has long double
    """
    if size == 4:
        typ = ir.FloatType()
    elif size == 8:
        typ = ir.DoubleType()
    else:
        typ = ir.DoubleType()
    return typ
 
def lift_insn(ida_insn: ida_hexrays.minsn_t, blk: ir.Block, builder: ir.IRBuilder) -> ir.Instruction:
    """
    This function lift microcode insn into llvm in following steps:
    1. Lift left, right and destination mop for each instruction.
    2. Lift instruction.

    ida_insn: microcode insn
    blk: current llvm block
    builder: llvm builder
    """
    builder.position_at_end(blk)
    l = lift_mop(ida_insn.l, blk, builder)
    # The load src is always address
    r = lift_mop(ida_insn.r, blk, builder, ida_insn.opcode == ida_hexrays.m_ldx)
    # The insn dest is always address except dest is call dest (arguments)
    d = lift_mop(ida_insn.d, blk, builder, True and ida_insn.opcode != ida_hexrays.m_call and ida_insn.opcode != ida_hexrays.m_icall)

    # create declaration for unknown intrinsic function
    if ida_insn.l.t == ida_hexrays.mop_h and l == None:
        l = create_intrinsic_function(blk.parent.parent, ida_insn.l.helper, ida_insn.d.f)

    blk_itr = iter(blk.parent.blocks)
    list(itertools.takewhile(lambda x: x.name != blk.name, blk_itr))
    # get next block
    next_blk = next(blk_itr, None)

    if ida_insn.opcode == ida_hexrays.m_nop:    # 0x00,  nop    no operation
        return
    elif ida_insn.opcode == ida_hexrays.m_stx:  # 0x01,  stx  l,    {r=sel, d=off}  store value to memory
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_ldx:  # 0x02,  ldx  {l=sel,r=off}, d load load value from memory
        if not ida_insn.is_fpinsn():            # maybe ldx.fpu
            typ = ir.IntType(ida_insn.d.size * 8)
        else:
            typ = float_type(ida_insn.d.size * 8)
        r = typecast(r, typ.as_pointer(), builder)  
        r = builder.load(r)
        d = storecast(r, d, builder)
        return _store_as(r, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_ldc:  # 0x03,  ldc  l=const,d   load constant
        r = ir.Constant(ir.IntType(32), ida_insn.l.nnn)
        return _store_as(r, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_mov:  # 0x04,  mov  l, d   move*F
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_neg:  # 0x05,  neg  l, d   negate
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        l = builder.neg(l)
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)    
    elif ida_insn.opcode == ida_hexrays.m_lnot:  # 0x06,  lnot l, d   logical not
        r = ir.IntType(ida_insn.l.size*8)(0)
        r = typecast(r, l.type, builder)  
        cmp = builder.icmp_unsigned("==", l, r)
        d = storecast(cmp, d, builder)
        return _store_as(cmp, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_bnot:  # 0x07,  bnot l, d   bitwise not
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        l = builder.not_(l)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_xds:  # 0x08,  xds  l, d   extend (signed)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_xdu:  # 0x09,  xdu  l, d   extend (unsigned)
        return _store_as(l, d, blk, builder, signed=False)
    elif ida_insn.opcode == ida_hexrays.m_low:  # 0x0A,  low  l, d   take low part
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_high:  # 0x0B,  high l, d   take high part
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_add:  # 0x0C,  add  l,   r, d   l + r -> dst
        if isinstance(l.type, ir.FloatType) or isinstance(l.type, ir.DoubleType):
            l = builder.fptoui(l, ir.IntType(ida_insn.l.size*8))
        if isinstance(r.type, ir.FloatType) or isinstance(r.type, ir.DoubleType):
            r = builder.fptoui(r, ir.IntType(ida_insn.r.size*8))

        if isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.IntType):
            castPtr = typecast(l, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(castPtr, (r, ))
            math = typecast(math, l.type, builder)
        elif isinstance(r.type, ir.PointerType) and isinstance(l.type, ir.IntType):
            castPtr = typecast(r, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(castPtr, (l, ))
            math = typecast(math, r.type, builder)
        elif isinstance(l.type, ir.IntType) and isinstance(r.type, ir.IntType):
            math = builder.add(l, r)
        elif isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.PointerType):
            ptrType = ir.IntType(64) # get pointer type
            const1 = builder.ptrtoint(l, ptrType)
            const2 = builder.ptrtoint(r, ptrType)
            math = builder.add(const1, const2)
        else:
            raise NotImplementedError("expected subtraction between pointer/integers")
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder) 
    elif ida_insn.opcode == ida_hexrays.m_sub:  # 0x0D,  sub  l,   r, d   l - r -> dst
        if isinstance(l.type, ir.FloatType) or isinstance(l.type, ir.DoubleType):
            l = builder.fptoui(l, ir.IntType(ida_insn.l.size*8))
        if isinstance(r.type, ir.FloatType) or isinstance(r.type, ir.DoubleType):
            r = builder.fptoui(r, ir.IntType(ida_insn.r.size*8))

        if isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.IntType):
            r = builder.neg(r)
            castPtr = typecast(l, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(castPtr, (r, ))
            math = typecast(math, l.type, builder)
        elif isinstance(r.type, ir.PointerType) and isinstance(l.type, ir.IntType):
            l = builder.neg(l)
            castPtr = typecast(r, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(castPtr, (l, ))
            math = typecast(math, r.type, builder)
        elif isinstance(l.type, ir.IntType) and isinstance(r.type, ir.IntType):
            math = builder.sub(l, r)
        elif isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.PointerType):
            ptrType = ir.IntType(64) # get pointer type
            const1 = builder.ptrtoint(l, ptrType)
            const2 = builder.ptrtoint(r, ptrType)
            math = builder.sub(const1, const2)
        else:
            raise NotImplementedError("expected subtraction between pointer/integers")
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder) 
    elif ida_insn.opcode == ida_hexrays.m_mul:  # 0x0E,  mul  l,   r, d   l * r -> dst
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.mul(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_udiv:  # 0x0F,  udiv l,   r, d   l / r -> dst
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.udiv(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_sdiv:  # 0x10,  sdiv l,   r, d   l / r -> dst
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        d = storecast(l, d, builder)
        math = builder.sdiv(l, r)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_umod:  # 0x11,  umod l,   r, d   l % r -> dst
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.urem(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_smod:  # 0x12,  smod l,   r, d   l % r -> dst
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.srem(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_or:  # 0x13,  or   l,   r, d   bitwise or
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.or_(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_and:  # 0x14,  and  l,   r, d   bitwise and
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.and_(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_xor:  # 0x15,  xor  l,   r, d   bitwise xor
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)
        math = builder.xor(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_shl:  # 0x16,  shl  l,   r, d   shift logical left
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)        
        math = builder.shl(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_shr:  # 0x17,  shr  l,   r, d   shift logical right
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)        
        math = builder.ashr(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_sar:  # 0x18,  sar  l,   r, d   shift arithmetic right
        l = typecast(l, ir.IntType(ida_insn.d.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size*8), builder)        
        math = builder.ashr(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_cfadd:  # 0x19,  cfadd l,  r,    d=carry    calculate carry    bit of (l+r)
        l = typecast(l, ir.IntType(64), builder)
        r = typecast(r, ir.IntType(64), builder)
        math = builder.call(lift_intrinsic_function(blk.parent.parent, "sadd_overflow"), [l, r])
        math = builder.gep(math, (ir.IntType(32)(0), ir.IntType(32)(0), ))
        math = builder.load(math)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_ofadd:  # 0x1A,  ofadd l,  r,    d=overf    calculate overflow bit of (l+r)
        l = typecast(l, ir.IntType(64), builder)
        r = typecast(r, ir.IntType(64), builder)
        math = builder.call(lift_intrinsic_function(blk.parent.parent, "sadd_overflow"), [l, r])
        math = builder.gep(math, (ir.IntType(32)(0), ir.IntType(32)(1),))
        math = builder.load(math)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_cfshl:  # 0x1B,  cfshl l,  r,    d=carry    calculate carry    bit of (l<<r)
        l = typecast(l, ir.IntType(64), builder)
        r = typecast(r, ir.IntType(64), builder)
        func_name = f"m_cfshr_{ida_insn.d.size}"
        if func_name in blk.parent.parent.globals:
            f_func = blk.parent.parent.get_global(func_name)
        else:
            f_func = blk.parent.parent.declare_intrinsic(func_name, fnty=ir.FunctionType(ir.DoubleType(), [ir.IntType(64), ir.IntType(64)]))
        l = builder.call(f_func, [l, r])
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_cfshr:  # 0x1C,  cfshr l,  r,    d=carry    calculate carry    bit of (l>>r)
        l = typecast(l, ir.IntType(64), builder)
        r = typecast(r, ir.IntType(64), builder)
        func_name = f"m_cfshr_{ida_insn.d.size}"
        if func_name in blk.parent.parent.globals:
            f_func = blk.parent.parent.get_global(func_name)
        else:
            f_func = blk.parent.parent.declare_intrinsic(func_name, fnty=ir.FunctionType(ir.DoubleType(), [ir.IntType(64), ir.IntType(64)]))
        l = builder.call(f_func, [l, r])
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_sets:  # 0x1D,  sets  l,d=byte  SF=1Sign
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = ir.Constant(ir.IntType(ida_insn.l.size*8), 0)
        cond = builder.icmp_unsigned("<", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_seto:  # 0x1E,  seto  l,  r, d=byte  OF=1Overflow of (l-r)
        l = typecast(l, ir.IntType(64), builder)
        r = typecast(r, ir.IntType(64), builder) 
        math = builder.call(lift_intrinsic_function(blk.parent.parent, "__OFSUB__"), [l, r])
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setp:  # 0x1F,  setp  l,  r, d=byte  PF=1Unordered/Parity  *F
        func_name = f"setp_{ida_insn.l.size}_{ida_insn.d.size}"
        if func_name in blk.parent.parent.globals:
            f_setp = blk.parent.parent.get_global(func_name)
        else:
            f_setp = blk.parent.parent.declare_intrinsic(func_name, fnty=ir.FunctionType(ir.IntType(ida_insn.d.size*8), [ir.IntType(ida_insn.l.size*8), ir.IntType(32)]))
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        l = builder.call(f_setp, [l, ir.Constant(ir.IntType(32), ida_insn.d.size)])
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setnz:  # 0x20,  setnz l,  r, d=byte  ZF=0Not Equal    *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("!=", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setz:  # 0x21,  setz  l,  r, d=byte  ZF=1Equal   *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("==", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setae:  # 0x22,  setae l,  r, d=byte  CF=0Above or Equal    *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned(">=", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setb:  # 0x23,  setb  l,  r, d=byte  CF=1Below   *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("<", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_seta:  # 0x24,  seta  l,  r, d=byte  CF=0 & ZF=0   Above   *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned(">", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setbe:  # 0x25,  setbe l,  r, d=byte  CF=1 | ZF=1   Below or Equal    *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("<=", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setg:  # 0x26,  setg  l,  r, d=byte  SF=OF & ZF=0  Greater
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed(">", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setge:  # 0x27,  setge l,  r, d=byte  SF=OF    Greater or Equal
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed(">=", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setl:  # 0x28,  setl  l,  r, d=byte  SF!=OF   Less
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed("<", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_setle:  # 0x29,  setle l,  r, d=byte  SF!=OF | ZF=1 Less or Equal
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed("<=", l, r)
        result = builder.select(cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0))
        return _store_as(result, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_jcnd:  # 0x2A,  jcnd   l,    d   d is mop_v or mop_b
        l = typecast(l, ir.IntType(1), builder)
        return builder.cbranch(l, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jnz:  # 0x2B,  jnz    l, r, d   ZF=0Not Equal *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("!=", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jz:  # 0x2C,  jzl, r, d   ZF=1Equal*F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("==", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jae:  # 0x2D,  jae    l, r, d   CF=0Above or Equal *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder)  
        cond = builder.icmp_unsigned(">=", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jb:  # 0x2E,  jbl, r, d   CF=1Below*F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("<", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_ja:  # 0x2F,  jal, r, d   CF=0 & ZF=0   Above*F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder)  
        cond = builder.icmp_unsigned(">", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jbe:  # 0x30,  jbe    l, r, d   CF=1 | ZF=1   Below or Equal *F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_unsigned("<=", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jg:  # 0x31,  jgl, r, d   SF=OF & ZF=0  Greater
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed(">", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jge:  # 0x32,  jge    l, r, d   SF=OF    Greater or Equal
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed(">=", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jl:  # 0x33,  jll, r, d   SF!=OF   Less
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed("<", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jle:  # 0x34,  jle    l, r, d   SF!=OF | ZF=1 Less or Equal
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        r = typecast(r, ir.IntType(ida_insn.r.size*8), builder) 
        cond = builder.icmp_signed("<=", l, r)
        return builder.cbranch(cond, d, next_blk)
    elif ida_insn.opcode == ida_hexrays.m_jtbl:  # 0x35,  jtbl   l, r=mcases    Table jump
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        if "default" in r:
            switch = builder.switch(l, blk.parent.basic_blocks[r["default"]])
        else:
            switch = builder.switch(l, blk.parent.basic_blocks[r[list(r.keys())[0]]])
        for value in r.keys():
            if isinstance(value, int):
                switch.add_case(value, blk.parent.basic_blocks[r[value]])
        return switch
    elif ida_insn.opcode == ida_hexrays.m_ijmp:  # 0x36,  ijmp  {r=sel, d=off}  indirect unconditional jump
        return
    elif ida_insn.opcode == ida_hexrays.m_goto:  # 0x37,  goto   l    l is mop_v or mop_b
        return builder.branch(l)
    elif ida_insn.opcode == ida_hexrays.m_call:  # 0x38,  call   ld   l is mop_v or mop_b or mop_h
        rets, args = d
        if not isinstance(l.type, ir.PointerType) or not isinstance(l.type.pointee, ir.FunctionType):
            argtype = []
            for (i, arg) in enumerate(args):
                argtype.append(arg.type)

            new_func_type = ir.FunctionType(i8ptr, argtype, var_arg=False).as_pointer()
            l = typecast(l, new_func_type, builder)

            ret = builder.call(l, args)
            for dst in rets:
                _store_as(ret, dst, blk, builder) 
            return ret

        for (i, llvmtype) in enumerate(l.type.pointee.args):
            if i >= len(args):
                args.append(ir.Constant(ir.IntType(32), 1))
            if args[i].type != llvmtype:
                args[i] = typecast(args[i], llvmtype, builder)
        
        args = args[:len(l.type.pointee.args)]
        
        if l.type.pointee.var_arg: # function is variadic
            ltype = l.type.pointee
            newargs = list(ltype.args)
            for i in range(len(newargs), len(args)):
                newargs.append(args[i].type)
            new_func_type = ir.FunctionType(ltype.return_type, newargs, var_arg=True).as_pointer()
            l = typecast(l, new_func_type, builder)
        ret = builder.call(l, args)
        for dst in rets:
            _store_as(ret, dst, blk, builder) 
        return ret
    elif ida_insn.opcode == ida_hexrays.m_icall:  # 0x39,  icall  {l=sel, r=off} d    indirect call
        rets, args = d
        ftype = ir.FunctionType(ir.IntType(8).as_pointer(), (arg.type for arg in args))
        f = typecast(r, ftype.as_pointer(), builder)
        return builder.call(f, args)
    elif ida_insn.opcode == ida_hexrays.m_ret:  # 0x3A,  ret
        return
    elif ida_insn.opcode == ida_hexrays.m_push:  # 0x3B,  push   l
        return
    elif ida_insn.opcode == ida_hexrays.m_pop:  # 0x3C,  popd
        return
    elif ida_insn.opcode == ida_hexrays.m_und:  # 0x3D,  undd   undefine
        return
    elif ida_insn.opcode == ida_hexrays.m_ext:  # 0x3E,  ext  in1, in2,  out1  external insn, not microcode *F
        return
    elif ida_insn.opcode == ida_hexrays.m_f2i:  # 0x3F,  f2il,    d int(l) => d; convert fp -> integer   +F
        return typecast(l, ir.IntType(ida_insn.d.size * 8), builder, signed=True)
    elif ida_insn.opcode == ida_hexrays.m_f2u:  # 0x40,  f2ul,    d uint(l)=> d; convert fp -> uinteger  +F
        return typecast(l, ir.IntType(ida_insn.d.size * 8), builder, signed=False)
    elif ida_insn.opcode == ida_hexrays.m_i2f:  # 0x41,  i2fl,    d fp(l)  => d; convert integer -> fp e +F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        if ida_insn.d.size == 4:
            typ = ir.FloatType()
        elif ida_insn.d.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType()      
        return builder.sitofp(l, typ)
    elif ida_insn.opcode == ida_hexrays.m_u2f:  # 0x42,  i2fl,    d fp(l)  => d; convert uinteger -> fp  +F
        l = typecast(l, ir.IntType(ida_insn.l.size*8), builder)
        if ida_insn.d.size == 4:
            typ = ir.FloatType()
        elif ida_insn.d.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType()   
        return builder.uitofp(l, typ)
    elif ida_insn.opcode == ida_hexrays.m_f2f:  # 0x43,  f2fl,    d l => d; change fp precision+F
        if ida_insn.d.size == 4:
            l = typecast(l, ir.FloatType(), builder)
            if d != None and d.type != ir.FloatType().as_pointer():
                d = typecast(d, ir.FloatType().as_pointer(), builder)

        if ida_insn.d.size == 8:
            l = typecast(l, ir.DoubleType(), builder)
            if d != None and d.type != ir.DoubleType().as_pointer():
                d = typecast(d, ir.DoubleType().as_pointer(), builder)

        if ida_insn.d.size == 16:
            l = typecast(l, ir.DoubleType(), builder)
            if d != None and d.type != ir.DoubleType().as_pointer():
                d = typecast(d, ir.DoubleType().as_pointer(), builder)
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_fneg:  # 0x44,  fneg    l,    d -l=> d; change sign   +F
        return _store_as(l, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_fadd:  # 0x45,  fadd    l, r, d l + r  => d; add +F
        if ida_insn.l.size == 4:
            typ = ir.FloatType()
        elif ida_insn.l.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType() 
        
        l = typecast(l, typ, builder)
        r = typecast(r, typ, builder)
        math = builder.fadd(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_fsub:  # 0x46,  fsub    l, r, d l - r  => d; subtract +F
        if ida_insn.l.size == 4:
            typ = ir.FloatType()
        elif ida_insn.l.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType() 
        
        l = typecast(l, typ, builder)
        r = typecast(r, typ, builder)
        math = builder.fsub(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_fmul:  # 0x47,  fmul    l, r, d l * r  => d; multiply +F
        if ida_insn.l.size == 4:
            typ = ir.FloatType()
        elif ida_insn.l.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType() 
        
        l = typecast(l, typ, builder)
        r = typecast(r, typ, builder)
        math = builder.fmul(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    elif ida_insn.opcode == ida_hexrays.m_fdiv:  # 0x48,  fdiv    l, r, d l / r  => d; divide   +F
        if ida_insn.l.size == 4:
            typ = ir.FloatType()
        elif ida_insn.l.size == 8:
            typ = ir.DoubleType()
        else:
            typ = ir.DoubleType() 
        
        l = typecast(l, typ, builder)
        r = typecast(r, typ, builder)
        math = builder.fdiv(l, r)
        d = storecast(l, d, builder)
        return _store_as(math, d, blk, builder)
    else:
        raise NotImplementedError(f"not implemented {ida_insn.dstr()}")

class BIN2LLVMController():
    """
    The control component of BinaryLift Explorer.
    """
    def __init__(self):
        self.m = ir.Module()

    def insertAllFunctions(self):
        """
        This function lift all text functions into llvm.
        """
        for f_ea in idautils.Functions():
            if idc.get_segm_name(f_ea) not in [".text"]:
                continue
            self.insertFunctionAtEa(f_ea)

    def insertFunctionAtEa(self, ea):
        """
        This function lift specific function into llvm.
        """
        if ea in ptext:
            typ = ptext[ea].type
            func_name = ida_name.get_name(ea)
            lift_function(self.m, func_name, False, ea, typ)

    def create_global(self, ea, width, str_dict):
        """
        This function create all known global variables in following steps.
        1. get data item name and type.
        2. create global variables.

        ea: data address
        width: data width in ea
        str_dict: all known strings
        """
        # get item name and get type information
        name = ida_name.get_name(ea)
        if name == "":
            name = f"data_{hex(ea)[2:]}"

        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, ea):
            ida_typeinf.guess_tinfo(tif, ea)
        
        # if data item is string, create string global variable.
        if ea in str_dict:
            str_csnt = str_dict[ea][0]
            strType = ir.ArrayType(ir.IntType(8), str_dict[ea][1])
            g = ir.GlobalVariable(self.m, strType, name=name)
            g.initializer = ir.Constant(strType, bytearray(str_csnt))
            g.linkage = "private"
            g.global_constant = True 

        # if data item is extern function, create declaration.
        elif tif.is_func() or tif.is_funcptr():
            if tif.is_funcptr():
                tif = tif.get_ptrarr_object()
            # if function is a thunk function, define the actual function instead
            if ((ida_funcs.get_func(ea) is not None)
            and (ida_funcs.get_func(ea).flags & ida_funcs.FUNC_THUNK)): 
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(ida_funcs.get_func(ea))
                if tfunc_ea != ida_idaapi.BADADDR:
                    ea = tfunc_ea
                    name = ida_name.get_name(ea)
            
            # create definition for declaration function,
            if ((ida_funcs.get_func(ea) is None)
            # or if the function is a library function,
            or (ida_funcs.get_func(ea).flags & ida_funcs.FUNC_LIB) 
            # or if the function is declared in a XTRN segment,
            or ida_segment.segtype(ea) & ida_segment.SEG_XTRN): 
                # return function declaration
                g = lift_function(self.m, name, True, ea, tif)

        # others data item, maybe int/float/arrary/structure                
        else:
            typ = lift_tif(tif, width)
            g_cmt = lift_from_address(self.m, ea, typ)
            g = ir.GlobalVariable(self.m, typ, name = name)
            g.initializer = g_cmt

    def initialize(self):
        """
        This function serves as a initial function with following steps.
        1. Decompile all functions.
        2. Collect all strings.
        3. Create GlobalVariabel for all IDA data item.

        ptext: dict to save decompile results {ea:decompile}
        str_dict: dict to save all strings
        """
        # decompile all functions
        for func in idautils.Functions():
            try:
                pfunc = ida_hexrays.decompile(func)
                if pfunc != None:
                    ptext[func] = pfunc
            except:
                pass

        # collect all strings identified by IDA
        str_dict = {}
        for s in list(idautils.Strings()):
            str_dict[s.ea] = [ida_bytes.get_bytes(s.ea, s.length), s.length]

        # iterative all data item and create global variable
        heads = list(idautils.Heads())
        for i in range(len(heads) - 1):
            start = heads[i]
            segm = idaapi.getseg(heads[i])
            if segm.perm & idaapi.SEGPERM_EXEC == 0:
                # set data boundary
                end = segm.end_ea if heads[i+1] > segm.end_ea else heads[i+1]
                self.create_global(start, end - start, str_dict)

    def save_to_file(self, filename):
        """
        This function save IR module with text format.
        """
        with open(filename, 'w') as f:
            f.write(str(self.m))

if __name__ == "__main__":
    """
    This script run in IDApython.
    output: The text IR file to be saved.
    """
    idc.auto_wait()
    output = idc.ARGV[1]
    bin2llvm = BIN2LLVMController()
    bin2llvm.initialize()
    bin2llvm.insertAllFunctions()
    bin2llvm.save_to_file(output)
    idc.qexit(0)

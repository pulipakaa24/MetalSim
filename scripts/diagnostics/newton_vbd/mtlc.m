// Compile a Metal source file (e.g. a Warp kernel-cache .metal) with the Warp fork's compile options and print the
// full error log, which Warp truncates. Build: clang -fobjc-arc -framework Metal -framework Foundation mtlc.m -o mtlc
// Run through scripts/gpu_run.sh: mtlc ~/Library/Caches/warp/<ver>/<module>/<module>.metal

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_2; o.mathMode = MTLMathModeSafe; o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    NSError* err = nil;
    NSDate* t0 = [NSDate date];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    printf("lib=%p time=%.1fs\n", lib, -[t0 timeIntervalSinceNow]);
    if (err) {
      for (NSString* l in [err.localizedDescription componentsSeparatedByString:@"\n"])
        if (![l containsString:@"warning:"] ) printf("%s\n", l.UTF8String);
      printf("userInfo: %s\n", [[err.userInfo description] substringToIndex:MIN((NSUInteger)3000, [[err.userInfo description] length])].UTF8String);
    }
  }
  return 0;
}
